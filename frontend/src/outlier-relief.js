// Outlier-aware vertical exaggeration.
//
// Vertical exaggeration scales every height by the same factor, so on a flat tile
// a lone hilltop, tower or pit (tens of metres against a few metres of relief)
// turns into a spire or a well as the slider goes up, and dwarfs the terrain the
// user is trying to see. Only such points are limited; the rest scales exactly as
// before.
//
// Which points: those beyond Tukey's fences, Q1 − 1.5·IQR and Q3 + 1.5·IQR, on
// the tile's own heights. That's where a height stops belonging to the tile's
// terrain, and where it starts running away from the rest under exaggeration.
// On real tiles this is about 1% of points on flat ground: Vidisha's hill above
// 445 m (0.8%, relief 408–476 m), Kolkata's towers (1.4%), Bardhaman (0.9%).
// Darjeeling is untouched (0%), since its heights are spread out, not spiked.
// The wider 3·IQR "extreme" fence was tried first. It caught only Vidisha's
// summit, and at ×50 the hill's slopes still grew into a spire.
//
// How: the part of a point beyond its fence gets a soft ceiling. Its displayed
// excess e·s (s = display scale) becomes C·tanh(e·s / C). Just past the fence
// that's linear, so a small excess is left alone. It bends as the excess nears C
// and never goes beyond C, however far the slider goes. C is one IQR (the
// tile's typical spread, 10 m on Vidisha) as shown at the automatic exaggeration.
// The terrain in between keeps scaling with s, so outliers shrink in proportion.
// A cap of half the fenced band's height was tried first. On Vidisha at ×50 it
// shortened the hill's spire by only ~30%.

const FENCE_K = 1.5;

function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// Fences for `values`, or null when nothing lies outside them.
export function outlierFences(values) {
    if (!values.length) {
        return null;
    }
    const sorted = Float64Array.from(values).sort();
    const q1 = quantile(sorted, 0.25);
    const q3 = quantile(sorted, 0.75);
    const iqr = q3 - q1;
    const lo = Math.max(sorted[0], q1 - FENCE_K * iqr);
    const hi = Math.min(sorted[sorted.length - 1], q3 + FENCE_K * iqr);
    if (!(iqr > 0) || (sorted[0] >= lo && sorted[sorted.length - 1] <= hi)) {
        return null;
    }
    const outside = sorted[0] < lo || sorted[sorted.length - 1] > hi;
    return outside ? { lo, hi, cap: iqr } : null;
}

// Writes into `out` the local heights that, once the mesh is scaled by `scale`,
// show `src` with its outliers limited. Inside the fences out[i] = src[i].
export function limitOutliers(src, out, fences, scale) {
    if (!fences || !(scale > 0)) {
        out.set(src);
        return out;
    }
    const { lo, hi, cap } = fences;
    for (let i = 0; i < src.length; i++) {
        const z = src[i];
        if (z > hi) {
            out[i] = hi + (cap / scale) * Math.tanh(((z - hi) * scale) / cap);
        } else if (z < lo) {
            out[i] = lo - (cap / scale) * Math.tanh(((lo - z) * scale) / cap);
        } else {
            out[i] = z;
        }
    }
    return out;
}
