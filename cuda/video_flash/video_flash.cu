#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

struct Audit { unsigned int area,red_union,red_now,max_variation,max_range,min_y,max_y; float mean_y; };
struct ShadowAudit { unsigned int blue,magenta,pixels,max_spread; };
struct Context {
    int w,h,n,window,pos,frames; float rate;
    unsigned char *input,*output,*red_ring;
    float *minimum_y,*maximum_y,*last_y,*variation,*deltas;
    unsigned int *histogram;
    unsigned char *red_count; Audit *stats; ShadowAudit *shadow_stats;
};
static thread_local char error_message[256];
__constant__ float linear_lut[256];
static bool ok(cudaError_t e) { if(e==cudaSuccess)return true; snprintf(error_message,256,"%s",cudaGetErrorString(e));return false; }
#define API extern "C" __declspec(dllexport)
API const char* video_flash_error(){return error_message;}
API void video_flash_destroy(void* handle){
    auto c=(Context*)handle;if(!c)return;
    cudaFree(c->input);cudaFree(c->output);cudaFree(c->minimum_y);cudaFree(c->maximum_y);cudaFree(c->histogram);cudaFree(c->last_y);
    cudaFree(c->variation);cudaFree(c->deltas);cudaFree(c->red_ring);cudaFree(c->red_count);cudaFree(c->stats);cudaFree(c->shadow_stats);delete c;
}
API void* video_flash_create(int w,int h,int window,float rate){
    if(w<1||h<1||window<1||window>120||rate<=0)return nullptr;
    auto c=new Context{};c->w=w;c->h=h;c->n=w*h;c->window=window;c->rate=rate;
    size_t n=c->n;float table[256];for(int i=0;i<256;i++){float v=i/255.f;table[i]=v<=.04045f?v/12.92f:powf((v+.055f)/1.055f,2.4f);}
    if(!ok(cudaMemcpyToSymbol(linear_lut,table,sizeof(table)))||
       !ok(cudaMalloc(&c->input,n*3))||!ok(cudaMalloc(&c->output,n*3))||
       !ok(cudaMalloc(&c->minimum_y,n*sizeof(float)))||!ok(cudaMalloc(&c->maximum_y,n*sizeof(float)))||
       !ok(cudaMalloc(&c->histogram,512*sizeof(unsigned int)))||!ok(cudaMalloc(&c->last_y,n*sizeof(float)))||
       !ok(cudaMalloc(&c->variation,n*sizeof(float)))||!ok(cudaMalloc(&c->deltas,n*window*sizeof(float)))||
       !ok(cudaMalloc(&c->red_ring,n*(window+1)))||!ok(cudaMalloc(&c->red_count,n))||!ok(cudaMalloc(&c->stats,sizeof(Audit)))||
       !ok(cudaMalloc(&c->shadow_stats,sizeof(ShadowAudit)))){
        video_flash_destroy(c);return nullptr;
    }
    cudaMemset(c->variation,0,n*sizeof(float));cudaMemset(c->deltas,0,n*window*sizeof(float));
    cudaMemset(c->red_ring,0,n*(window+1));cudaMemset(c->red_count,0,n);return c;
}
__device__ unsigned char encode(float v){v=fminf(1.f,fmaxf(0.f,v));float x=v<=.0031308f?12.92f*v:1.055f*powf(v,1.f/2.4f)-.055f;return (unsigned char)__float2int_rn(255.f*x);}
__global__ void histogram_frame(Context c){
    __shared__ unsigned int bins[512];int t=threadIdx.x;bins[t]=0;bins[t+256]=0;__syncthreads();
    int i=blockIdx.x*blockDim.x+t;
    if(i<c.n){int x=(i%c.w)*1280/c.w,y=(i/c.w)*960/c.h;
        if(x>=20&&x<980&&y>=70&&y<790){
            float v=.2126f*linear_lut[c.input[3*i]]+.7152f*linear_lut[c.input[3*i+1]]+.0722f*linear_lut[c.input[3*i+2]];
            int j=min(511,max(0,(int)floorf((log2f(fmaxf(v,1.f/65536.f))+16.f)*32.f)));
            atomicAdd(&bins[j],1);
        }
    }
    __syncthreads();atomicAdd(&c.histogram[t],bins[t]);atomicAdd(&c.histogram[t+256],bins[t+256]);
}
__global__ void process_frame(Context c,float exposure,float span){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.n)return;
    int x=(i%c.w)*1280/c.w,y=(i/c.w)*960/c.h;
    bool panel=(x>=20&&x<980&&y>=70&&y<790);
    if(!panel){for(int k=0;k<3;k++)c.output[3*i+k]=c.input[3*i+k];return;}
    float3 q=make_float3(linear_lut[c.input[3*i]],linear_lut[c.input[3*i+1]],linear_lut[c.input[3*i+2]]);
    float y0=.2126f*q.x+.7152f*q.y+.0722f*q.z;
    float t=fminf(1.f,fmaxf(0.f,.5f+(log2f(fmaxf(y0,1.f/65536.f))-exposure)*.22f));
    float target=.008f+span*t;
    // A per-frame tone curve, never a mix with another frame. Preserve luminance
    // while retaining 85% of chroma; desaturate saturated red and out-of-gamut colors.
    float saturation=.85f;
    float excess=q.x-.60f*(q.x+q.y+q.z);
    if(excess>0)saturation=fminf(saturation,1.f-excess/(excess+.80f*y0));
    // Lift shadows with neutral light, never by amplifying their chroma. Tiny
    // source/codec differences such as RGB(0,1,6) must not become bright blue.
    // When darkening, scale chroma with luminance to retain the source hue.
    float gain=fminf(1.f,target/fmaxf(y0,1e-12f));
    float3 chroma=make_float3((q.x-y0)*gain,(q.y-y0)*gain,(q.z-y0)*gain);
    float top=fmaxf(chroma.x,fmaxf(chroma.y,chroma.z));
    if(top>0)saturation=fminf(saturation,(1.f-target)/top);
    if(y0==0)chroma=make_float3(0,0,0);
    q=make_float3(target+saturation*chroma.x,target+saturation*chroma.y,target+saturation*chroma.z);
    unsigned char r=encode(q.x),g=encode(q.y),b=encode(q.z);
    // Quantizing very dark colors must not turn a desaturated color into red-only RGB(1,0,0).
    for(int k=0;k<3;k++){
        float sum=linear_lut[r]+linear_lut[g]+linear_lut[b];
        if(sum<=0||linear_lut[r]<.72f*sum)break;
        if(g<255)g++;if(b<255)b++;
    }
    c.output[3*i]=r;c.output[3*i+1]=g;c.output[3*i+2]=b;
}
__global__ void process_levels(Context c,float black,float white,float gamma,float floor,float span,float saturation0){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.n)return;
    int x=(i%c.w)*1280/c.w,y=(i/c.w)*960/c.h;
    bool panel=(x>=20&&x<980&&y>=70&&y<790);
    if(!panel){for(int k=0;k<3;k++)c.output[3*i+k]=c.input[3*i+k];return;}
    float3 q=make_float3(linear_lut[c.input[3*i]],linear_lut[c.input[3*i+1]],linear_lut[c.input[3*i+2]]);
    float y0=.2126f*q.x+.7152f*q.y+.0722f*q.z;
    // Current-frame black/white levels: every output stays within [floor,floor+span].
    float t=powf(fminf(1.f,fmaxf(0.f,(y0-black)/(white-black))),gamma);
    float target=floor+span*t;
    float saturation=saturation0;
    float excess=q.x-.60f*(q.x+q.y+q.z);
    if(excess>0)saturation=fminf(saturation,1.f-excess/(excess+.80f*y0));
    // Chroma may follow a lift only where the source has real signal; near-black
    // codec noise such as RGB(0,1,6) is lifted neutrally, never amplified.
    float limit=1.f+2.f*fminf(1.f,fmaxf(0.f,(y0-.001f)/.009f));
    float gain=fminf(limit,target/fmaxf(y0,1e-12f));
    float3 chroma=make_float3((q.x-y0)*gain,(q.y-y0)*gain,(q.z-y0)*gain);
    float top=fmaxf(chroma.x,fmaxf(chroma.y,chroma.z));
    if(top>0)saturation=fminf(saturation,(1.f-target)/top);
    float bottom=fminf(chroma.x,fminf(chroma.y,chroma.z));
    if(bottom<0)saturation=fminf(saturation,target/-bottom);
    if(y0==0)chroma=make_float3(0,0,0);
    q=make_float3(target+saturation*chroma.x,target+saturation*chroma.y,target+saturation*chroma.z);
    unsigned char r=encode(q.x),g=encode(q.y),b=encode(q.z);
    for(int k=0;k<3;k++){
        float sum=linear_lut[r]+linear_lut[g]+linear_lut[b];
        if(sum<=0||linear_lut[r]<.72f*sum)break;
        if(g<255)g++;if(b<255)b++;
    }
    c.output[3*i]=r;c.output[3*i+1]=g;c.output[3*i+2]=b;
}
__global__ void audit_shadows(Context c){
    int i=blockIdx.x*blockDim.x+threadIdx.x;unsigned int blue=0,magenta=0,pixels=0,spread=0;
    if(i<c.n){int x=(i%c.w)*1280/c.w,y=(i/c.w)*960/c.h;
        if(x>=20&&x<980&&y>=70&&y<790){
            int input_max=max((int)c.input[3*i],max((int)c.input[3*i+1],(int)c.input[3*i+2]));
            if(input_max<16){int r=c.output[3*i],g=c.output[3*i+1],b=c.output[3*i+2];pixels=1;
                blue=b>r+35&&b>g+35;magenta=r>g+30&&b>g+30;spread=max(r,max(g,b))-min(r,min(g,b));}
        }
    }
    __shared__ unsigned int bs[256],ms[256],ps[256],ss[256];int t=threadIdx.x;
    bs[t]=blue;ms[t]=magenta;ps[t]=pixels;ss[t]=spread;__syncthreads();
    for(int s=128;s;s/=2){if(t<s){bs[t]+=bs[t+s];ms[t]+=ms[t+s];ps[t]+=ps[t+s];ss[t]=max(ss[t],ss[t+s]);}__syncthreads();}
    if(t==0){atomicAdd(&c.shadow_stats->blue,bs[0]);atomicAdd(&c.shadow_stats->magenta,ms[0]);atomicAdd(&c.shadow_stats->pixels,ps[0]);atomicMax(&c.shadow_stats->max_spread,ss[0]);}
}
API int video_flash_shadow_check(void* handle,unsigned int* output){
    auto c=(Context*)handle;if(!c)return -1;
    if(!ok(cudaMemset(c->shadow_stats,0,sizeof(ShadowAudit))))return -1;
    audit_shadows<<<(c->n+255)/256,256>>>(*c);
    return ok(cudaGetLastError())&&ok(cudaMemcpy(output,c->shadow_stats,sizeof(ShadowAudit),cudaMemcpyDeviceToHost))?0:-1;
}
__device__ float lin(float v){v=fminf(1.f,fmaxf(0.f,v));return v<=.04045f?v/12.92f:powf((v+.055f)/1.055f,2.4f);}
__global__ void audit_frame(Context c,int yuv){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int a=0,ru=0,rn=0,m=0,range=0,ybits=0;
    if(i<c.n){
        float r,g,b;
        if(yuv){int x=i%c.w,y=i/c.w,uv=(y/2)*(c.w/2)+x/2;float yy=(c.input[i]-16.f)/219.f;
            float cb=(c.input[c.n+uv]-128.f)/224.f,cr=(c.input[c.n+c.n/4+uv]-128.f)/224.f;
            r=lin(yy+1.5748f*cr);g=lin(yy-.187324f*cb-.468124f*cr);b=lin(yy+1.8556f*cb);
        }else{r=linear_lut[c.input[3*i]];g=linear_lut[c.input[3*i+1]];b=linear_lut[c.input[3*i+2]];}
        float y=.2126f*r+.7152f*g+.0722f*b;
        float low=c.frames?fminf(c.minimum_y[i],y):y,high=c.frames?fmaxf(c.maximum_y[i],y):y;
        c.minimum_y[i]=low;c.maximum_y[i]=high;range=__float_as_uint(high-low);ybits=__float_as_uint(y);
        float delta=c.frames?fabsf(y-c.last_y[i]):0.f;c.last_y[i]=y;
        size_t j=(size_t)(c.pos%c.window)*c.n+i;
        float sum=fmaxf(0.f,c.variation[i]+delta-c.deltas[j]);c.deltas[j]=delta;c.variation[i]=sum;
        rn=(r+g+b)>0&&r/(r+g+b)>=.8f;
        size_t k=(size_t)(c.pos%(c.window+1))*c.n+i;
        int count=(int)c.red_count[i]+(int)rn-(int)c.red_ring[k];c.red_count[i]=(unsigned char)count;c.red_ring[k]=(unsigned char)rn;
        // A general flash needs opposing luminance changes >=0.10. An all-time
        // pixel range <0.10 excludes it, even if many tiny changes sum to >=0.55.
        // All-time range is a conservative superset of every one-second range.
        a=sum>=.55f&&(high-low)>=.10f;ru=count>0;m=__float_as_uint(sum);
    }
    __shared__ unsigned int area[256],red[256],now[256],peak[256],ranges[256],lo[256],hi[256];__shared__ float means[256];int t=threadIdx.x;
    means[t]=__uint_as_float(ybits);
    area[t]=a;red[t]=ru;now[t]=rn;peak[t]=m;ranges[t]=range;lo[t]=i<c.n?ybits:0xffffffff;hi[t]=ybits;__syncthreads();
    for(int s=128;s;s/=2){if(t<s){area[t]+=area[t+s];red[t]+=red[t+s];now[t]+=now[t+s];peak[t]=max(peak[t],peak[t+s]);ranges[t]=max(ranges[t],ranges[t+s]);lo[t]=min(lo[t],lo[t+s]);hi[t]=max(hi[t],hi[t+s]);means[t]+=means[t+s];}__syncthreads();}
    if(t==0){atomicAdd(&c.stats->area,area[0]);atomicAdd(&c.stats->red_union,red[0]);atomicAdd(&c.stats->red_now,now[0]);atomicMax(&c.stats->max_variation,peak[0]);atomicMax(&c.stats->max_range,ranges[0]);atomicMin(&c.stats->min_y,lo[0]);atomicMax(&c.stats->max_y,hi[0]);atomicAdd(&c.stats->mean_y,means[0]/c.n);}
}
API int video_flash_histogram(void* handle,const unsigned char* input,unsigned int* output){
    auto c=(Context*)handle;if(!c)return -1;
    if(!ok(cudaMemcpy(c->input,input,c->n*3,cudaMemcpyHostToDevice))||!ok(cudaMemset(c->histogram,0,512*sizeof(unsigned int))))return -1;
    histogram_frame<<<(c->n+255)/256,256>>>(*c);
    return ok(cudaGetLastError())&&ok(cudaMemcpy(output,c->histogram,512*sizeof(unsigned int),cudaMemcpyDeviceToHost))?0:-1;
}
API int video_flash_process(void* handle,const unsigned char* input,unsigned char* output,float exposure,float span){
    auto c=(Context*)handle;if(!c)return -1;
    if(!ok(cudaMemcpy(c->input,input,c->n*3,cudaMemcpyHostToDevice)))return -1;
    process_frame<<<(c->n+255)/256,256>>>(*c,exposure,span);
    return ok(cudaGetLastError())&&ok(cudaMemcpy(output,c->output,c->n*3,cudaMemcpyDeviceToHost))?0:-1;
}
API int video_flash_process_levels(void* handle,const unsigned char* input,unsigned char* output,float black,float white,float gamma,float floor,float span,float saturation){
    auto c=(Context*)handle;if(!c||!(white>black)||gamma<=0||floor<0||span<=0)return -1;
    if(!ok(cudaMemcpy(c->input,input,c->n*3,cudaMemcpyHostToDevice)))return -1;
    process_levels<<<(c->n+255)/256,256>>>(*c,black,white,gamma,floor,span,saturation);
    return ok(cudaGetLastError())&&ok(cudaMemcpy(output,c->output,c->n*3,cudaMemcpyDeviceToHost))?0:-1;
}
static int audit_host(void* handle,const unsigned char* input,float* output,int yuv){
    auto c=(Context*)handle;if(!c)return -1;
    if(!ok(cudaMemcpy(c->input,input,yuv?c->n*3/2:c->n*3,cudaMemcpyHostToDevice))||!ok(cudaMemset(c->stats,0,sizeof(Audit)))||!ok(cudaMemset(&c->stats->min_y,255,sizeof(unsigned int))))return -1;
    audit_frame<<<(c->n+255)/256,256>>>(*c,yuv);Audit s;
    if(!ok(cudaGetLastError())||!ok(cudaMemcpy(&s,c->stats,sizeof(s),cudaMemcpyDeviceToHost)))return -1;
    float peak;memcpy(&peak,&s.max_variation,sizeof(float));output[0]=peak;output[1]=(float)s.area;output[2]=(float)s.red_union;output[3]=(float)s.red_now;
    memcpy(output+4,&s.max_range,sizeof(float));memcpy(output+5,&s.min_y,sizeof(float));memcpy(output+6,&s.max_y,sizeof(float));
    output[7]=s.mean_y;
    c->pos=(c->pos+1)%(c->window*(c->window+1));c->frames++;return 0;
}
API int video_flash_audit(void* h,const unsigned char* p,float* out){return audit_host(h,p,out,0);}
API int video_flash_audit_yuv420(void* h,const unsigned char* p,float* out){return audit_host(h,p,out,1);}
