// Map style for the video's OSM panel: Protomaps basemap layers (BSD-3), dark
// flavour recoloured to the video's navy UI. Data: (c) OpenStreetMap contributors.
// Shared by the renderer page; `basemaps` and `maplibregl` are globals.
const PALETTE = {
  background: '#0d2133',
  water: '#0d2133',
  earth: '#27343e',
  glacier: '#3a4852',
  sand: '#2f3a3f', beach: '#2f3a3f',
  wood_a: '#25363a', wood_b: '#25363a',
  park_a: '#26373a', park_b: '#26373a',
  scrub_a: '#28373a', scrub_b: '#28373a',
  boundaries: '#8496a4',
  city_label: '#eef3f7', city_label_halo: '#101c26',
  state_label: '#9fb2c1', state_label_halo: '#101c26',
  country_label: '#c6d3dd',
  ocean_label: '#7fa8c2',
  waterway_label: '#7fa8c2',
};

// One clean label per feature: English name, else the local name.
function englishLabels(layers) {
  const name = ['coalesce', ['get', 'name:en'], ['get', 'name']];
  return layers.map((layer) => {
    const field = layer.layout && layer.layout['text-field'];
    if (field && JSON.stringify(field).includes('name')) return { ...layer, layout: { ...layer.layout, 'text-field': name } };
    return layer;
  });
}

function buildStyle(origin, flavorName = 'dark', overrides = PALETTE) {
  const flavor = { ...basemaps.namedFlavor(flavorName), ...overrides };
  return {
    version: 8,
    glyphs: `${origin}/assets/fonts/{fontstack}/{range}.pbf`,
    sprite: `${origin}/assets/sprites/v4/${flavorName}`,
    sources: {
      protomaps: {
        type: 'vector',
        tiles: [`${origin}/tiles/{z}/{x}/{y}.mvt`],
        maxzoom: 12,
        attribution: '<a href="https://protomaps.com">Protomaps</a> © <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>',
      },
    },
    layers: englishLabels(basemaps.layers('protomaps', flavor, { lang: 'en' })),
  };
}
