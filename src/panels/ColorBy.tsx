/**
 * What the map is coloured by.
 *
 * This control is new, but nothing under it is: `store/scene.ts` has carried `colorBy`
 * since Phase 7, `Shell.tsx` fetches the column for it, `scene/colors.ts` picks a ramp per
 * feature and `PointCloud` rewrites 150,000 floats when it changes. All of that shipped
 * with no way to reach it from the interface. For a map whose entire job is to make a
 * library legible at a glance, "colour it by brightness" is the single most useful thing
 * the app can be asked to do, so it gets a button on the top bar.
 *
 * The list is curated, not the `Feature` enum. The enum is thirteen DSP columns, several of
 * which are confidences for other columns, and a menu of `bpmConfidence` and `zeroCrossing`
 * is a menu written for the person who implemented the analyser. These are the seven a
 * producer would actually reach for, under the names they would look for them by.
 */

import type { Feature } from '../ipc';
import { useSceneStore } from '../store/scene';
import { ChoiceRow, PanelHeading } from './Chrome';

interface Choice {
  /** `null` restores the colours the projection fit produced. */
  feature: Feature | null;
  label: string;
  /** What the ramp means, in the words someone browsing samples would use. */
  hint: string;
}

const CHOICES: readonly Choice[] = [
  { feature: null, label: 'Similarity', hint: 'the fit' },
  { feature: 'spectralCentroid', label: 'Brightness', hint: 'dark to bright' },
  { feature: 'lufsIntegrated', label: 'Loudness', hint: 'LUFS' },
  { feature: 'bpm', label: 'Tempo', hint: 'BPM' },
  { feature: 'keyRoot', label: 'Key', hint: 'C to B' },
  { feature: 'spectralFlatness', label: 'Noisiness', hint: 'tonal to noisy' },
  { feature: 'onsetDensity', label: 'Busyness', hint: 'onsets per second' },
  { feature: 'durationMs', label: 'Length', hint: 'short to long' },
];

/** Content for the shell's shared contextual area. */
export function ColorByPanel({ onChoose }: { onChoose?: () => void }) {
  const colorBy = useSceneStore((s) => s.colorBy);
  const setColorBy = useSceneStore((s) => s.setColorBy);

  return (
    <section>
      <PanelHeading>Colour by</PanelHeading>
      <div className="space-y-0.5">
        {CHOICES.map((choice) => (
          <ChoiceRow
            key={choice.label}
            selected={colorBy === choice.feature}
            onClick={() => {
              setColorBy(choice.feature);
              onChoose?.();
            }}
          >
            <span>{choice.label}</span>
            {/* Inherits the row's colour rather than setting its own, so the hint stays
                readable against the accent fill when the row is the current pick. */}
            <span className="font-mono text-[10px] opacity-55">{choice.hint}</span>
          </ChoiceRow>
        ))}
      </div>
    </section>
  );
}
