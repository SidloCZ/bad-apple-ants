#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define API extern "C" __declspec(dllexport)
static thread_local std::string last_error;
static void check(cudaError_t code) { if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
static void blas(cublasStatus_t code) { if(code!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error "+std::to_string(code)); }
static void fft_check(cufftResult code) { if(code!=CUFFT_SUCCESS) throw std::runtime_error("cuFFT error "+std::to_string(code)); }
template<class T> struct Device {
    T* ptr=nullptr; size_t count=0;
    Device()=default; explicit Device(size_t n) { allocate(n); }
    Device(const Device&)=delete; Device& operator=(const Device&)=delete;
    ~Device(){ if(ptr) cudaFree(ptr); }
    void allocate(size_t n) { if(n<=count)return; if(ptr)check(cudaFree(ptr)); ptr=nullptr; count=0; if(n)check(cudaMalloc((void**)&ptr,n*sizeof(T))); count=n; }
    void upload(const T* host,size_t n) { allocate(n); check(cudaMemcpy(ptr,host,n*sizeof(T),cudaMemcpyHostToDevice)); }
    void download(T* host,size_t n) { check(cudaMemcpy(host,ptr,n*sizeof(T),cudaMemcpyDeviceToHost)); }
};
struct Tile {
    const uint8_t* bytes=nullptr; uint32_t w=0,h=0; uint64_t data_start=0;
    double left=0,bottom=0,right=0,top=0;
};
struct Affine { double a,b,c,d,e,f; };
static Affine transform(const double* geo,int w,int h,double aspect) {
    const double radians=geo[3]*(3.14159265358979323846/180.);
    const double s=sin(radians),c=cos(radians),dx=geo[2]/w;
    const double longitude_scale=111.32*cos(geo[1]*(3.14159265358979323846/180.));
    if(aspect<=0 || aspect==double(h)/w) {
        return {c*dx/longitude_scale,-s*dx/longitude_scale,
            geo[0]+(-c*w/2.+s*h/2.)*dx/longitude_scale,
            -s*dx/111.32,-c*dx/111.32,geo[1]+(s*w/2.+c*h/2.)*dx/111.32};
    }
    const double dy=geo[2]*aspect/h;
    Affine t{c*dx/longitude_scale,-s*dy/longitude_scale,0,-s*dx/111.32,-c*dy/111.32,0};
    t.c=geo[0]-t.a*w/2.-t.b*h/2.;t.f=geo[1]-t.d*w/2.-t.e*h/2.;return t;
}
__device__ uint8_t native_value(const Tile* tiles,double lon,double lat) {
    if(!(lon>=-180. && lon<180. && lat>=-90. && lat<=90.))return 255;
    const int col=int(floor((lon+180.)/30.)),row=int(floor((90.-lat)/30.));
    if(col<0||col>=12||row<0||row>=6)return 255;
    const Tile t=tiles[row*12+col];if(!t.bytes)return 255;
    const double sx=1./((t.right-t.left)/t.w),sy=-1./((t.top-t.bottom)/t.h);
    const int x=int(floor(lon*sx-t.left*sx)),y=int(floor(lat*sy-t.top*sy));
    if(x<0||x>=int(t.w)||y<0||y>=int(t.h))return 255;
    const auto* offsets=reinterpret_cast<const uint64_t*>(t.bytes+16);
    const auto* ends=reinterpret_cast<const uint32_t*>(t.bytes+t.data_start);
    uint64_t start=offsets[y],lo=start,hi=offsets[y+1];
    while(lo<hi){uint64_t mid=(lo+hi)/2;if(ends[mid]<=uint32_t(x))lo=mid+1;else hi=mid;}
    return uint8_t((lo-start)&1);
}
__global__ void sample_kernel(const Tile* tiles,const Affine* matrices,int n,int w,int h,uint8_t* output,const int* mirrors=nullptr) {
    const size_t pixel=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int view=blockIdx.y;if(view>=n||pixel>=size_t(w)*h)return;
    const Affine t=matrices[view];double x=double(pixel%w)+.5;const double y=double(pixel/w)+.5;
    if(mirrors&&mirrors[view])x=w-x;
    output[size_t(view)*w*h+pixel]=native_value(tiles,t.a*x+t.b*y+t.c,t.d*x+t.e*y+t.f);
}
__global__ void feature_kernel(const Tile* tiles,const Affine* matrices,int n,int w,int h,int aa,int8_t* output,uint32_t* valid) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int dimension=w*h,view=int(i/dimension);if(view>=n)return;
    const int x=int(i%dimension)%w,y=int(i%dimension)/w;const Affine t=matrices[view];int mass=0;
    for(int yy=0;yy<aa;++yy)for(int xx=0;xx<aa;++xx){
        const double px=x+(xx+.5)/aa,py=y+(yy+.5)/aa;
        const uint8_t value=native_value(tiles,t.a*px+t.b*py+t.c,t.d*px+t.e*py+t.f);
        if(value==255)atomicExch(valid+view,0u);else mass+=value;
    }
    output[i]=int8_t(mass);
}
__global__ void statistics_kernel(const int8_t* features,const uint8_t* valid,int rows,int dimension,float2* stats) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    if(valid&&!valid[row]){stats[row]=make_float2(0,0);return;}
    int sum=0,squares=0;for(int j=0;j<dimension;++j){const int v=features[size_t(row)*dimension+j];sum+=v;squares+=v*v;}
    const float variance=fmaxf(0.f,float(squares)-float(sum)*sum/dimension);
    stats[row]=make_float2(float(sum),variance>1e-5f?rsqrtf(variance):0.f);
}
__global__ void initialize_best(float* scores,int* indices,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count){scores[i]=-CUDART_INF_F;indices[i]=-1;}
}
__global__ void rank_kernel(const int32_t* products,const float2* queries,const float2* bank,int n,int dimension,
    int offset,int k,float* best_scores,int* best_indices) {
    const int q=blockIdx.x,tid=threadIdx.x;float values[8];int indices[8];
    for(int j=0;j<8;++j){values[j]=-CUDART_INF_F;indices[j]=-1;}
    const float2 qs=queries[q];
    for(int i=tid;i<n;i+=blockDim.x){
        const float2 bs=bank[i];if(qs.y==0||bs.y==0)continue;
        const float score=(float(products[size_t(q)*n+i])-qs.x*bs.x/dimension)*qs.y*bs.y;
        if(score<=values[k-1])continue;
        int at=k-1;while(at>0&&score>values[at-1]){values[at]=values[at-1];indices[at]=indices[at-1];--at;}
        values[at]=score;indices[at]=offset+i;
    }
    __shared__ float scores[256];__shared__ int ids[256];
    for(int j=0;j<k;++j){
        scores[tid]=values[0];ids[tid]=indices[0];__syncthreads();
        for(int stride=128;stride;stride/=2){if(tid<stride){
            const float other=scores[tid+stride];const int oi=ids[tid+stride];
            if(other>scores[tid]||(other==scores[tid]&&oi>=0&&(ids[tid]<0||oi<ids[tid]))){scores[tid]=other;ids[tid]=oi;}
        }__syncthreads();}
        const float selected=scores[0];const int selected_id=ids[0];
        if(tid==0&&selected_id>=0){
            float* out=best_scores+size_t(q)*k;int* out_i=best_indices+size_t(q)*k;
            int at=k;for(int l=0;l<k;++l)if(selected>out[l]||(selected==out[l]&&(out_i[l]<0||selected_id<out_i[l]))){at=l;break;}
            if(at<k){for(int l=k-1;l>at;--l){out[l]=out[l-1];out_i[l]=out_i[l-1];}out[at]=selected;out_i[at]=selected_id;}
        }
        if(indices[0]==selected_id&&selected_id>=0){for(int l=0;l<k-1;++l){values[l]=values[l+1];indices[l]=indices[l+1];}values[k-1]=-CUDART_INF_F;indices[k-1]=-1;}
        __syncthreads();
    }
}
struct Context {
    std::array<Tile,72> host_tiles{};Device<Tile> tiles;
    Device<Affine> transforms;Device<uint8_t> masks,valid_bytes;Device<uint32_t> flags;
    Device<int8_t> feature_bytes,query_features;Device<float2> feature_stats,query_stats;
    Device<int32_t> products;Device<float> best_scores;Device<int> best_indices;
    cublasHandle_t handle=nullptr;int query_count=0,dimension=0,top_k=0;
    Device<float> fft_scene,fft_query,fft_output;Device<cufftComplex> scene_spectrum,query_spectrum;
    Device<float2> integral;Device<float> refined;
    Device<uint8_t> exact_query,query_edge,query_dilated,scene_edge,scene_dilated;
    Device<unsigned long long> counts;
    Device<uint16_t> distance_a,distance_b;
    Device<uint32_t> shape_weights;
    Device<unsigned long long> shape_counts,border_counts;
    Device<int> mirror_flags;
    cufftHandle forward=0,inverse=0,query_forward=0;int fft_w=0,fft_h=0,fft_batch=0;
    Context(){check(cudaSetDevice(0));blas(cublasCreate(&handle));tiles.allocate(72);}
    ~Context(){if(forward)cufftDestroy(forward);if(inverse)cufftDestroy(inverse);if(query_forward)cufftDestroy(query_forward);for(auto& tile:host_tiles)if(tile.bytes)cudaFree((void*)tile.bytes);if(handle)cublasDestroy(handle);}
};
template<class F>static int guarded(F&& body){try{body();last_error.clear();return 0;}catch(const std::exception& e){last_error=e.what();return -1;}}
API const char* coast_cuda_error(){return last_error.c_str();}
API void* coast_cuda_create(){try{auto* c=new Context();last_error.clear();return c;}catch(const std::exception& e){last_error=e.what();return nullptr;}}
API void coast_cuda_destroy(void* handle){delete static_cast<Context*>(handle);}
API int coast_cuda_add_tile(void* handle,int index,const char* path,double left,double bottom,double right,double top){return guarded([&]{
    if(!handle||index<0||index>=72)throw std::runtime_error("Invalid native tile slot");auto& c=*static_cast<Context*>(handle);
    // UTF-8 path from Python; u8path keeps non-ASCII folders working on Windows.
    std::ifstream input(std::filesystem::u8path(path),std::ios::binary|std::ios::ate);if(!input)throw std::runtime_error("Cannot open native RLE tile");
    const auto size=size_t(input.tellg());input.seekg(0);std::vector<uint8_t> bytes(size);
    if(!input.read((char*)bytes.data(),size))throw std::runtime_error("Cannot read native RLE tile");
    if(size<16||std::string((char*)bytes.data(),8)!="LWMRLE01")throw std::runtime_error("Invalid native RLE header");
    uint32_t w,h;memcpy(&w,bytes.data()+8,4);memcpy(&h,bytes.data()+12,4);const size_t data_start=16+8*(size_t(h)+1);
    if(!w||!h||size<data_start)throw std::runtime_error("Truncated native RLE row index");
    uint64_t last;memcpy(&last,bytes.data()+16+size_t(h)*8,8);if(data_start+last*4!=size)throw std::runtime_error("Truncated native RLE transitions");
    // The device binary-searches each row's transitions, so the row offsets must be sane.
    uint64_t previous=0;for(size_t row=0;row<=h;++row){uint64_t offset;memcpy(&offset,bytes.data()+16+row*8,8);
        if(offset<previous||offset>last)throw std::runtime_error("Corrupt native RLE row index");previous=offset;}
    uint8_t* ptr=nullptr;check(cudaMalloc((void**)&ptr,size));check(cudaMemcpy(ptr,bytes.data(),size,cudaMemcpyHostToDevice));
    if(c.host_tiles[index].bytes)check(cudaFree((void*)c.host_tiles[index].bytes));
    c.host_tiles[index]={ptr,w,h,uint64_t(data_start),left,bottom,right,top};
});}
API int coast_cuda_finish_tiles(void* handle){return guarded([&]{auto& c=*static_cast<Context*>(handle);for(const auto& t:c.host_tiles)if(!t.bytes)throw std::runtime_error("Incomplete original-resolution globe");c.tiles.upload(c.host_tiles.data(),72);});}
static void upload_transforms(Context& c,const double* geometry,int n,int w,int h,double aspect){
    if(n<=0||w<=0||h<=0)throw std::runtime_error("Invalid native sampling dimensions");std::vector<Affine> matrices(n);
    for(int i=0;i<n;++i){if(!(geometry[i*4+2]>0))throw std::runtime_error("Invalid view width");matrices[i]=transform(geometry+i*4,w,h,aspect);}c.transforms.upload(matrices.data(),n);
}
static const int* upload_mirrors(Context& c,const int* mirrors,int n){if(!mirrors)return nullptr;c.mirror_flags.upload(mirrors,n);return c.mirror_flags.ptr;}
static int sample_impl(void* handle,const double* geometry,int n,int w,int h,double aspect,uint8_t* output,const int* mirrors=nullptr){return guarded([&]{
    auto& c=*static_cast<Context*>(handle);upload_transforms(c,geometry,n,w,h,aspect);c.masks.allocate(size_t(n)*w*h);
    sample_kernel<<<dim3((w*h+255)/256,n),256>>>(c.tiles.ptr,c.transforms.ptr,n,w,h,c.masks.ptr,upload_mirrors(c,mirrors,n));check(cudaGetLastError());c.masks.download(output,size_t(n)*w*h);
});}
API int coast_cuda_sample(void* handle,const double* geometry,int n,int w,int h,double aspect,uint8_t* output){return sample_impl(handle,geometry,n,w,h,aspect,output);}
API int coast_cuda_sample_mirrored(void* handle,const double* geometry,const int* mirrors,int n,int w,int h,double aspect,uint8_t* output){return sample_impl(handle,geometry,n,w,h,aspect,output,mirrors);}
API int coast_cuda_features(void* handle,const double* geometry,int n,int w,int h,int aa,double aspect,int8_t* output,uint8_t* valid){return guarded([&]{
    if(aa<1||aa>11)throw std::runtime_error("Native feature samples do not fit int8");auto& c=*static_cast<Context*>(handle);
    upload_transforms(c,geometry,n,w,h,aspect);c.feature_bytes.allocate(size_t(n)*w*h);c.flags.allocate(n);
    std::vector<uint32_t> flags(n,1);c.flags.upload(flags.data(),n);
    feature_kernel<<<unsigned((size_t(n)*w*h+255)/256),256>>>(c.tiles.ptr,c.transforms.ptr,n,w,h,aa,c.feature_bytes.ptr,c.flags.ptr);
    check(cudaGetLastError());c.feature_bytes.download(output,size_t(n)*w*h);c.flags.download(flags.data(),n);for(int i=0;i<n;++i)valid[i]=uint8_t(flags[i]);
});}
API int coast_cuda_queries(void* handle,const int8_t* queries,int n,int dimension){return guarded([&]{
    auto& c=*static_cast<Context*>(handle);if(n<=0||dimension<=0||dimension%4)throw std::runtime_error("Invalid cuBLAS descriptor shape");
    c.query_count=n;c.dimension=dimension;c.query_features.upload(queries,size_t(n)*dimension);c.query_stats.allocate(n);
    statistics_kernel<<<(n+127)/128,128>>>(c.query_features.ptr,nullptr,n,dimension,c.query_stats.ptr);check(cudaGetLastError());check(cudaDeviceSynchronize());
});}
API int coast_cuda_start_region(void* handle,int k){return guarded([&]{auto& c=*static_cast<Context*>(handle);if(k<1||k>8||!c.query_count)throw std::runtime_error("Invalid regional top-k");
    c.top_k=k;size_t n=size_t(c.query_count)*k;c.best_scores.allocate(n);c.best_indices.allocate(n);initialize_best<<<unsigned((n+255)/256),256>>>(c.best_scores.ptr,c.best_indices.ptr,n);check(cudaGetLastError());
});}
static void rank_device(Context& c,int n,int offset){
    if(!c.top_k||n<=0)throw std::runtime_error("Regional search not initialized");
    c.feature_stats.allocate(n);c.products.allocate(size_t(n)*c.query_count);
    statistics_kernel<<<(n+127)/128,128>>>(c.feature_bytes.ptr,c.valid_bytes.ptr,n,c.dimension,c.feature_stats.ptr);
    const int32_t alpha=1,beta=0;
    blas(cublasGemmEx(c.handle,CUBLAS_OP_T,CUBLAS_OP_N,n,c.query_count,c.dimension,&alpha,
        c.feature_bytes.ptr,CUDA_R_8I,c.dimension,c.query_features.ptr,CUDA_R_8I,c.dimension,&beta,
        c.products.ptr,CUDA_R_32I,n,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    rank_kernel<<<c.query_count,256>>>(c.products.ptr,c.query_stats.ptr,c.feature_stats.ptr,n,c.dimension,offset,c.top_k,c.best_scores.ptr,c.best_indices.ptr);
    check(cudaGetLastError());
}
API int coast_cuda_rank(void* handle,const int8_t* bank,const uint8_t* valid,int n,int offset){return guarded([&]{
    auto& c=*static_cast<Context*>(handle);c.feature_bytes.upload(bank,size_t(n)*c.dimension);c.valid_bytes.upload(valid,n);
    rank_device(c,n,offset);check(cudaDeviceSynchronize());
});}
API int coast_cuda_region_results(void* handle,float* scores,int* indices){return guarded([&]{auto& c=*static_cast<Context*>(handle);size_t n=size_t(c.query_count)*c.top_k;c.best_scores.download(scores,n);c.best_indices.download(indices,n);});}

__global__ void flags_to_bytes(const uint32_t* flags,uint8_t* valid,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)valid[i]=flags[i]!=0;}
__global__ void contour_descriptor(const int8_t* input,const float2* stats,int8_t* output,int n,int w,int h){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;int dimension=w*h,view=int(i/dimension);if(view>=n)return;
    int x=int(i%dimension)%w,y=int(i%dimension)/w,value=input[i],mean=__float2int_rn(stats[view].x/dimension);
    size_t base=size_t(view)*dimension*3,at=i%dimension;
    output[base+at]=int8_t(value-mean);output[base+dimension+at]=int8_t(x?2*(value-int(input[i-1])):0);
    output[base+2*dimension+at]=int8_t(y?2*(value-int(input[i-w])):0);
}
API int coast_cuda_native_rank_profiles(void* area_handle,void* contour_handle,const double* geometry,int n,int w,int h,int aa,double aspect,int offset,int area,int contours){return guarded([&]{
    auto& c=*static_cast<Context*>(area_handle);auto& d=*static_cast<Context*>(contour_handle);
    if(n<1||aa<1||aa>7||c.dimension!=w*h||d.dimension!=3*w*h)throw std::runtime_error("Invalid resident descriptor dimensions");
    upload_transforms(c,geometry,n,w,h,aspect);c.feature_bytes.allocate(size_t(n)*w*h);c.flags.allocate(n);c.valid_bytes.allocate(n);
    std::vector<uint32_t> flags(n,1);c.flags.upload(flags.data(),n);
    feature_kernel<<<unsigned((size_t(n)*w*h+255)/256),256>>>(c.tiles.ptr,c.transforms.ptr,n,w,h,aa,c.feature_bytes.ptr,c.flags.ptr);
    flags_to_bytes<<<(n+255)/256,256>>>(c.flags.ptr,c.valid_bytes.ptr,n);
    if(contours){
        c.feature_stats.allocate(n);statistics_kernel<<<(n+127)/128,128>>>(c.feature_bytes.ptr,c.valid_bytes.ptr,n,w*h,c.feature_stats.ptr);
        d.feature_bytes.allocate(size_t(n)*3*w*h);d.valid_bytes.allocate(n);
        contour_descriptor<<<unsigned((size_t(n)*w*h+255)/256),256>>>(c.feature_bytes.ptr,c.feature_stats.ptr,d.feature_bytes.ptr,n,w,h);
        check(cudaMemcpy(d.valid_bytes.ptr,c.valid_bytes.ptr,n,cudaMemcpyDeviceToDevice));
    }
    if(area)rank_device(c,n,offset);if(contours)rank_device(d,n,offset);check(cudaDeviceSynchronize());
});}

static int smooth_size(int n){for(int i=n;;++i){int k=i;for(int p:{2,3,5,7})while(k%p==0)k/=p;if(k==1)return i;}}
static void plans(Context& c,int w,int h,int n){
    if(c.fft_w==w&&c.fft_h==h&&c.fft_batch==n)return;
    if(c.forward)cufftDestroy(c.forward);if(c.inverse)cufftDestroy(c.inverse);if(c.query_forward)cufftDestroy(c.query_forward);
    c.forward=c.inverse=c.query_forward=0;int dimensions[2]={h,w};
    fft_check(cufftPlanMany(&c.forward,2,dimensions,nullptr,1,w*h,nullptr,1,h*(w/2+1),CUFFT_R2C,n));
    fft_check(cufftPlanMany(&c.inverse,2,dimensions,nullptr,1,h*(w/2+1),nullptr,1,w*h,CUFFT_C2R,n));
    fft_check(cufftPlan2d(&c.query_forward,h,w,CUFFT_R2C));c.fft_w=w;c.fft_h=h;c.fft_batch=n;
}
__global__ void prepare_fft(const uint8_t* masks,float* padded,int n,int w,int h,int fw,int fh){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(n)*fw*fh;if(i>=total)return;
    int x=i%fw,y=(i/fw)%fh,view=i/(size_t(fw)*fh);float v=0;
    if(x<w&&y<h){uint8_t b=masks[(size_t(view)*h+y)*w+x];v=b==255?0.f:float(b);}padded[i]=v;
}
__global__ void spectrum_product(cufftComplex* scenes,const cufftComplex* query,size_t per,size_t total){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=total)return;
    cufftComplex a=scenes[i],b=query[i%per];scenes[i]=make_cuFloatComplex(a.x*b.x+a.y*b.y,a.y*b.x-a.x*b.y);
}
__global__ void row_integral(const uint8_t* masks,float2* integral,int n,int w,int h){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n*h)return;int view=row/h,y=row%h;
    size_t dst=(size_t(view)*(h+1)+y+1)*(w+1);float land=0,missing=0;
    for(int x=0;x<w;++x){uint8_t v=masks[size_t(row)*w+x];land+=v==1;missing+=v==255;integral[dst+x+1]=make_float2(land,missing);}
}
__global__ void column_integral(float2* integral,int n,int w,int h){
    int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=n*w)return;int view=col/w,x=col%w;
    size_t at=size_t(view)*(h+1)*(w+1)+x+1;float2 sum=make_float2(0,0);
    for(int y=1;y<=h;++y){size_t i=at+size_t(y)*(w+1);float2 v=integral[i];sum.x+=v.x;sum.y+=v.y;integral[i]=sum;}
}
__device__ float2 rectangle(const float2* data,int stride,int x,int y,int w,int h){
    float2 a=data[size_t(y+h)*stride+x+w],b=data[size_t(y)*stride+x+w],c=data[size_t(y+h)*stride+x],d=data[size_t(y)*stride+x];
    return make_float2(a.x-b.x-c.x+d.x,a.y-b.y-c.y+d.y);
}
__global__ void best_translation(const float* correlations,const float2* integral,int sw,int sh,int fw,int fh,int w,int h,float qsum,float* output,int k,int separation){
    int view=blockIdx.x,tid=threadIdx.x,nx=sw-w+1,ny=sh-h+1;float best=-1.f;int best_i=0;
    const float2* sums=integral+size_t(view)*(sw+1)*(sh+1);const float* corr=correlations+size_t(view)*fw*fh;
    __shared__ float scores[256];__shared__ int ids[256];__shared__ int selected[8];
    for(int rank=0;rank<k;++rank){best=-1.f;best_i=0;
    for(int i=tid;i<nx*ny;i+=blockDim.x){int x=i%nx,y=i/nx;float2 box=rectangle(sums,sw+1,x,y,w,h);if(box.y>.1f)continue;
        bool duplicate=false;for(int j=0;j<rank;++j)if(abs(x-selected[j]%nx)<=separation&&abs(y-selected[j]/nx)<=separation){duplicate=true;break;}if(duplicate)continue;
        const float intersection=fminf(fminf(qsum,box.x),fmaxf(0.f,corr[size_t(y)*fw+x]/(fw*fh)));
        float u=qsum+box.x-intersection,wu=float(w*h)-intersection;
        float score=.5f*((u>1e-5f?intersection/u:1.f)+(wu>1e-5f?(float(w*h)-qsum-box.x+intersection)/wu:1.f));
        if(score>best){best=score;best_i=i;}
    }
    scores[tid]=best;ids[tid]=best_i;__syncthreads();
    for(int s=128;s;s/=2){if(tid<s&&(scores[tid+s]>scores[tid]||(scores[tid+s]==scores[tid]&&ids[tid+s]<ids[tid]))){scores[tid]=scores[tid+s];ids[tid]=ids[tid+s];}__syncthreads();}
    if(tid==0){selected[rank]=ids[0];size_t at=(size_t(view)*k+rank)*3;output[at]=float(ids[0]%nx);output[at+1]=float(ids[0]/nx);output[at+2]=scores[0];}__syncthreads();
    }
}
static int refine_impl(void* handle,const double* geometry,int n,const float* query,int w,int h,int mx,int my,double aspect,float* output,int k,int separation,const int* mirrors=nullptr){return guarded([&]{
    if(n<1||n>4096||w<1||h<1||mx<0||my<0||k<1||k>8||separation<0)throw std::runtime_error("Invalid FFT refinement dimensions");
    auto& c=*static_cast<Context*>(handle);int sw=w+2*mx,sh=h+2*my,fw=smooth_size(sw),fh=smooth_size(sh);
    std::vector<Affine> transforms(n);for(int i=0;i<n;++i){auto t=transform(geometry+i*4,w,h,aspect);t.c-=t.a*mx+t.b*my;t.f-=t.d*mx+t.e*my;transforms[i]=t;}
    c.transforms.upload(transforms.data(),n);c.masks.allocate(size_t(n)*sw*sh);
    sample_kernel<<<dim3((sw*sh+255)/256,n),256>>>(c.tiles.ptr,c.transforms.ptr,n,sw,sh,c.masks.ptr,upload_mirrors(c,mirrors,n));
    plans(c,fw,fh,n);size_t real=size_t(fw)*fh,complex=size_t(fh)*(fw/2+1);
    c.fft_scene.allocate(real*n);c.fft_output.allocate(real*n);c.fft_query.allocate(real);
    c.scene_spectrum.allocate(complex*n);c.query_spectrum.allocate(complex);
    std::vector<float> padded(real,0.f);double qsum=0;for(int y=0;y<h;++y)for(int x=0;x<w;++x){float v=query[size_t(y)*w+x];if(v<0||v>1)throw std::runtime_error("Query outside 0..1");padded[size_t(y)*fw+x]=v;qsum+=v;}
    c.fft_query.upload(padded.data(),real);
    prepare_fft<<<unsigned((real*n+255)/256),256>>>(c.masks.ptr,c.fft_scene.ptr,n,sw,sh,fw,fh);
    fft_check(cufftExecR2C(c.forward,c.fft_scene.ptr,c.scene_spectrum.ptr));
    fft_check(cufftExecR2C(c.query_forward,c.fft_query.ptr,c.query_spectrum.ptr));
    spectrum_product<<<unsigned((complex*n+255)/256),256>>>(c.scene_spectrum.ptr,c.query_spectrum.ptr,complex,complex*n);
    fft_check(cufftExecC2R(c.inverse,c.scene_spectrum.ptr,c.fft_output.ptr));
    size_t ints=size_t(n)*(sw+1)*(sh+1);c.integral.allocate(ints);check(cudaMemset(c.integral.ptr,0,ints*sizeof(float2)));
    row_integral<<<(n*sh+127)/128,128>>>(c.masks.ptr,c.integral.ptr,n,sw,sh);
    column_integral<<<(n*sw+127)/128,128>>>(c.integral.ptr,n,sw,sh);c.refined.allocate(size_t(n)*k*3);
    best_translation<<<n,256>>>(c.fft_output.ptr,c.integral.ptr,sw,sh,fw,fh,w,h,float(qsum),c.refined.ptr,k,separation);
    check(cudaGetLastError());c.refined.download(output,size_t(n)*k*3);
});}
API int coast_cuda_refine(void* handle,const double* geometry,int n,const float* query,int w,int h,int mx,int my,double aspect,float* output){
    return refine_impl(handle,geometry,n,query,w,h,mx,my,aspect,output,1,0);
}
API int coast_cuda_refine_topk(void* handle,const double* geometry,int n,const float* query,int w,int h,int mx,int my,double aspect,int k,int separation,float* output){
    return refine_impl(handle,geometry,n,query,w,h,mx,my,aspect,output,k,separation);
}
API int coast_cuda_refine_mirrored(void* handle,const double* geometry,const int* mirrors,int n,const float* query,int w,int h,int mx,int my,double aspect,int k,int separation,float* output){
    return refine_impl(handle,geometry,n,query,w,h,mx,my,aspect,output,k,separation,mirrors);
}
__global__ void edge_kernel(const uint8_t* masks,uint8_t* edges,int n,int w,int h,int border_y){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,pixels=size_t(w)*h;if(i>=pixels*n)return;
    int x=i%w,y=(i/w)%h;uint8_t v=masks[i];bool e=false;
    if(v!=255&&y>=border_y&&y<h-border_y){if(x>0)e|=masks[i-1]!=255&&masks[i-1]!=v;if(x+1<w)e|=masks[i+1]!=255&&masks[i+1]!=v;
        if(y>border_y)e|=masks[i-w]!=255&&masks[i-w]!=v;if(y+1<h-border_y)e|=masks[i+w]!=255&&masks[i+w]!=v;}edges[i]=e;
}
__global__ void dilate_kernel(const uint8_t* edges,uint8_t* dilated,int n,int w,int h,int tolerance){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,pixels=size_t(w)*h;if(i>=pixels*n)return;
    int x=i%w,y=(i/w)%h;size_t base=(i/pixels)*pixels;bool found=false;
    for(int yy=max(0,y-tolerance);yy<=min(h-1,y+tolerance)&&!found;++yy)
        for(int xx=max(0,x-tolerance);xx<=min(w-1,x+tolerance);++xx)if(edges[base+size_t(yy)*w+xx]){found=true;break;}
    dilated[i]=found;
}
__global__ void score_kernel(const uint8_t* masks,const uint8_t* query,const uint8_t* se,const uint8_t* sd,const uint8_t* qe,const uint8_t* qd,int pixels,int n,int w,int h,int border_y,unsigned long long* counts){
    int view=blockIdx.y;size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(view>=n)return;
    unsigned int values[7]={0,0,0,0,0,0,0};
    if(p<pixels){size_t i=size_t(view)*pixels+p;
        const int y=int(p/w);const bool inside=y>=border_y&&y<h-border_y;
        values[0]=inside&&masks[i]==1;values[1]=inside&&masks[i]==1&&query[p];values[2]=masks[i]==255;
        values[3]=se[i];values[4]=qe[p];values[5]=se[i]&&qd[p];values[6]=qe[p]&&sd[i];}
    __shared__ unsigned int sums[7*8];
    for(int j=0;j<7;++j){unsigned int v=values[j];for(int offset=16;offset;offset/=2)v+=__shfl_down_sync(0xffffffff,v,offset);
        if((threadIdx.x&31)==0)sums[j*8+threadIdx.x/32]=v;}
    __syncthreads();
    if(threadIdx.x<7){unsigned int sum=0;for(int warp=0;warp<8;++warp)sum+=sums[threadIdx.x*8+warp];
        if(sum)atomicAdd(counts+size_t(view)*7+threadIdx.x,(unsigned long long)sum);}
}
static int score_impl(void* handle,const double* geometry,int n,const uint8_t* query,int w,int h,double aspect,int tolerance,unsigned long long* output,const int* mirrors=nullptr,int border_y=0){return guarded([&]{
    if(n<1||w<1||h<1||w*h%32||tolerance<0||tolerance>8||border_y<0||2*border_y>=h)throw std::runtime_error("Invalid full-resolution scoring dimensions");
    auto& c=*static_cast<Context*>(handle);size_t pixels=size_t(w)*h,total=pixels*n;
    upload_transforms(c,geometry,n,w,h,aspect);c.masks.allocate(total);c.exact_query.upload(query,pixels);
    c.query_edge.allocate(pixels);c.query_dilated.allocate(pixels);c.scene_edge.allocate(total);c.scene_dilated.allocate(total);
    sample_kernel<<<dim3((pixels+255)/256,n),256>>>(c.tiles.ptr,c.transforms.ptr,n,w,h,c.masks.ptr,upload_mirrors(c,mirrors,n));
    edge_kernel<<<unsigned((pixels+255)/256),256>>>(c.exact_query.ptr,c.query_edge.ptr,1,w,h,border_y);
    edge_kernel<<<unsigned((total+255)/256),256>>>(c.masks.ptr,c.scene_edge.ptr,n,w,h,border_y);
    dilate_kernel<<<unsigned((pixels+255)/256),256>>>(c.query_edge.ptr,c.query_dilated.ptr,1,w,h,tolerance);
    dilate_kernel<<<unsigned((total+255)/256),256>>>(c.scene_edge.ptr,c.scene_dilated.ptr,n,w,h,tolerance);
    c.counts.allocate(size_t(n)*7);check(cudaMemset(c.counts.ptr,0,size_t(n)*7*sizeof(unsigned long long)));
    score_kernel<<<dim3((pixels+255)/256,n),256>>>(c.masks.ptr,c.exact_query.ptr,c.scene_edge.ptr,c.scene_dilated.ptr,c.query_edge.ptr,c.query_dilated.ptr,int(pixels),n,w,h,border_y,c.counts.ptr);
    check(cudaGetLastError());c.counts.download(output,size_t(n)*7);
});}
API int coast_cuda_score(void* handle,const double* geometry,int n,const uint8_t* query,int w,int h,double aspect,int tolerance,unsigned long long* output){return score_impl(handle,geometry,n,query,w,h,aspect,tolerance,output);}

// Exact truncated Manhattan distance, using separable min-plus propagation.
// The resulting soft contour score is deterministic: sums use integer weights.
__global__ void distance_initialize(const uint8_t* scene,const uint8_t* query,uint16_t* out,size_t pixels,int n){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=pixels*(n+1))return;
    out[i]=(i<pixels*n?scene[i]:query[i-pixels*n])?0:256;
}
__global__ void distance_step(const uint16_t* input,uint16_t* output,int w,int h,int n,int step,bool vertical){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,pixels=size_t(w)*h;if(i>=pixels*n)return;
    int at=vertical?int((i/w)%h):int(i%w),length=vertical?h:w;size_t stride=vertical?w:1;
    int best=input[i];if(at>=step)best=min(best,int(input[i-size_t(step)*stride])+step);
    if(at+step<length)best=min(best,int(input[i+size_t(step)*stride])+step);output[i]=uint16_t(best);
}
__global__ void shape_count_kernel(const uint8_t* se,const uint8_t* qe,const uint16_t* distances,const uint32_t* weights,
    size_t pixels,int n,unsigned long long* counts){
    int view=blockIdx.y;size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(view>=n)return;
    unsigned values[6]={0,0,0,0,0,0};
    if(p<pixels){size_t i=size_t(view)*pixels+p;unsigned qd=distances[size_t(n)*pixels+p],sd=distances[i];
        values[0]=se[i]?weights[qd]:0;values[1]=qe[p]?weights[sd]:0;
        values[2]=se[i]?weights[257+qd]:0;values[3]=qe[p]?weights[257+sd]:0;values[4]=se[i]?qd:0;values[5]=qe[p]?sd:0;}
    __shared__ unsigned int sums[6*8];
    for(int j=0;j<6;++j){unsigned v=values[j];for(int offset=16;offset;offset/=2)v+=__shfl_down_sync(0xffffffff,v,offset);
        if((threadIdx.x&31)==0)sums[j*8+threadIdx.x/32]=v;}
    __syncthreads();
    if(threadIdx.x<6){unsigned int sum=0;for(int warp=0;warp<8;++warp)sum+=sums[threadIdx.x*8+warp];
        if(sum)atomicAdd(counts+size_t(view)*6+threadIdx.x,(unsigned long long)sum);}
}
static int shape_impl(void* handle,const double* geometry,int n,const uint8_t* query,int w,int h,double aspect,int tolerance,unsigned long long* output,const int* mirrors=nullptr,int border_y=0){
    if(n<1){last_error="Invalid view count";return -1;}
    std::vector<unsigned long long> base(size_t(n)*7);
    int status=score_impl(handle,geometry,n,query,w,h,aspect,tolerance,base.data(),mirrors,border_y);if(status)return status;
    return guarded([&]{
        auto& c=*static_cast<Context*>(handle);size_t pixels=size_t(w)*h,total=pixels*(n+1);
        c.distance_a.allocate(total);c.distance_b.allocate(total);
        distance_initialize<<<unsigned((total+255)/256),256>>>(c.scene_edge.ptr,c.query_edge.ptr,c.distance_a.ptr,pixels,n);
        uint16_t* src=c.distance_a.ptr;uint16_t* dst=c.distance_b.ptr;
        for(int axis=0;axis<2;++axis)for(int step=1;step<256;step*=2){
            distance_step<<<unsigned((total+255)/256),256>>>(src,dst,w,h,n+1,step,axis!=0);std::swap(src,dst);
        }
        std::vector<uint32_t> weights(514);for(int j=0;j<2;++j){double tau=(j?32.:8.)*w/1444.;
            for(int d=0;d<=256;++d)weights[j*257+d]=d==256?0:uint32_t(std::llround(1000000.*std::exp(-d/tau)));}
        c.shape_weights.upload(weights.data(),weights.size());c.shape_counts.allocate(size_t(n)*6);
        check(cudaMemset(c.shape_counts.ptr,0,size_t(n)*6*sizeof(unsigned long long)));
        shape_count_kernel<<<dim3((pixels+255)/256,n),256>>>(c.scene_edge.ptr,c.query_edge.ptr,src,c.shape_weights.ptr,pixels,n,c.shape_counts.ptr);
        check(cudaGetLastError());std::vector<unsigned long long> shape(size_t(n)*6);c.shape_counts.download(shape.data(),shape.size());
        for(int i=0;i<n;++i){std::copy_n(base.data()+size_t(i)*7,7,output+size_t(i)*13);std::copy_n(shape.data()+size_t(i)*6,6,output+size_t(i)*13+7);}
    });
}
API int coast_cuda_score_shape(void* handle,const double* geometry,int n,const uint8_t* query,int w,int h,double aspect,int tolerance,unsigned long long* output){return shape_impl(handle,geometry,n,query,w,h,aspect,tolerance,output);}
API int coast_cuda_score_shape_mirrored(void* handle,const double* geometry,const int* mirrors,int n,const uint8_t* query,int w,int h,double aspect,int tolerance,unsigned long long* output){return shape_impl(handle,geometry,n,query,w,h,aspect,tolerance,output,mirrors);}

// Compare the four genuine-content borders, not the source video's letterbox.
__global__ void content_border_kernel(const uint8_t* masks,const uint8_t* query,int n,int w,int h,int border_y,unsigned long long* counts){
    const int view=blockIdx.y/4,side=blockIdx.y%4,k=blockIdx.x*blockDim.x+threadIdx.x;
    const int height=h-2*border_y,length=side<2?w:height;unsigned value=0;
    if(view<n&&k<length){size_t p;
        if(side==0)p=size_t(border_y)*w+k;
        else if(side==1)p=size_t(h-border_y-1)*w+k;
        else p=size_t(k+border_y)*w+(side==2?0:w-1);
        value=masks[size_t(view)*w*h+p]!=query[p];
    }
    for(int offset=16;offset;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
    __shared__ unsigned sums[8];if((threadIdx.x&31)==0)sums[threadIdx.x/32]=value;__syncthreads();
    if(threadIdx.x==0){unsigned sum=0;for(int warp=0;warp<8;++warp)sum+=sums[warp];
        if(sum)atomicAdd(counts+size_t(view)*4+side,(unsigned long long)sum);}
}
API int coast_cuda_score_structure(void* handle,const double* geometry,const int* mirrors,int n,const uint8_t* query,int w,int h,double aspect,int tolerance,int border_y,unsigned long long* output){
    if(n<1){last_error="Invalid view count";return -1;}
    std::vector<unsigned long long> base(size_t(n)*13);
    const int status=shape_impl(handle,geometry,n,query,w,h,aspect,tolerance,base.data(),mirrors,border_y);if(status)return status;
    return guarded([&]{auto& c=*static_cast<Context*>(handle);c.border_counts.allocate(size_t(n)*4);
        check(cudaMemset(c.border_counts.ptr,0,size_t(n)*4*sizeof(unsigned long long)));
        content_border_kernel<<<dim3((std::max(w,h-2*border_y)+255)/256,n*4),256>>>(c.masks.ptr,c.exact_query.ptr,n,w,h,border_y,c.border_counts.ptr);
        check(cudaGetLastError());std::vector<unsigned long long> borders(size_t(n)*4);c.border_counts.download(borders.data(),borders.size());
        for(int i=0;i<n;++i){std::copy_n(base.data()+size_t(i)*13,13,output+size_t(i)*17);std::copy_n(borders.data()+size_t(i)*4,4,output+size_t(i)*17+13);}
    });
}
