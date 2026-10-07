export const KNOWN_LAYERS = {
  volume: { label: 'Combined solve', kind: 'scalar' },
  sigma: { label: 'Uncertainty (sigma)', kind: 'scalar' },
  snr: { label: 'SNR', kind: 'scalar' },
  views: { label: 'View count', kind: 'scalar' },
  rays: { label: 'Ray count', kind: 'scalar' },
};

export function availableLayers(layerNames) {
  return layerNames.map((key) => {
    const known = KNOWN_LAYERS[key];
    return known ? { key, label: known.label, kind: known.kind }
                 : { key, label: key, kind: 'scalar' };
  });
}
