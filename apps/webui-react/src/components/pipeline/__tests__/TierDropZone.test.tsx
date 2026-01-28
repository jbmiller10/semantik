import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import '@testing-library/jest-dom';

import { TierDropZone } from '../TierDropZone';

describe('TierDropZone', () => {
  const renderInSVG = (component: React.ReactElement) => {
    return render(<svg>{component}</svg>);
  };

  const defaultBounds = { x: 40, y: 220, width: 200, height: 80 };

  it('renders when active', () => {
    const { container } = renderInSVG(
      <TierDropZone
        tier="chunker"
        bounds={defaultBounds}
        isActive={true}
        isHovered={false}
      />
    );

    const group = container.querySelector('g.tier-drop-zone');
    expect(group).toBeInTheDocument();
  });

  it('does not render when inactive', () => {
    const { container } = renderInSVG(
      <TierDropZone
        tier="chunker"
        bounds={defaultBounds}
        isActive={false}
        isHovered={false}
      />
    );

    const group = container.querySelector('g.tier-drop-zone');
    expect(group).not.toBeInTheDocument();
  });

  it('renders correct label for each tier type', () => {
    const tiers = ['parser', 'chunker', 'extractor', 'embedder'] as const;
    const expectedLabels = ['+ Add parser', '+ Add chunker', '+ Add extractor', '+ Add embedder'];

    tiers.forEach((tier, index) => {
      const { container } = renderInSVG(
        <TierDropZone
          tier={tier}
          bounds={defaultBounds}
          isActive={true}
          isHovered={false}
        />
      );

      const text = container.querySelector('text');
      expect(text?.textContent).toBe(expectedLabels[index]);
    });
  });

  it('has dashed stroke when not hovered', () => {
    const { container } = renderInSVG(
      <TierDropZone
        tier="chunker"
        bounds={defaultBounds}
        isActive={true}
        isHovered={false}
      />
    );

    const rect = container.querySelector('rect');
    expect(rect).toHaveAttribute('stroke-dasharray', '8 4');
  });

  it('has solid stroke when hovered', () => {
    const { container } = renderInSVG(
      <TierDropZone
        tier="chunker"
        bounds={defaultBounds}
        isActive={true}
        isHovered={true}
      />
    );

    const rect = container.querySelector('rect');
    expect(rect).toHaveAttribute('stroke-dasharray', 'none');
  });

  it('has background fill when hovered', () => {
    const { container } = renderInSVG(
      <TierDropZone
        tier="chunker"
        bounds={defaultBounds}
        isActive={true}
        isHovered={true}
      />
    );

    const rect = container.querySelector('rect');
    expect(rect).toHaveAttribute('fill', 'var(--bg-tertiary)');
    expect(rect).toHaveAttribute('fill-opacity', '0.5');
  });

  it('has no fill when not hovered', () => {
    const { container } = renderInSVG(
      <TierDropZone
        tier="chunker"
        bounds={defaultBounds}
        isActive={true}
        isHovered={false}
      />
    );

    const rect = container.querySelector('rect');
    expect(rect).toHaveAttribute('fill', 'none');
  });

  it('positions rect at provided bounds', () => {
    const bounds = { x: 100, y: 200, width: 300, height: 100 };
    const { container } = renderInSVG(
      <TierDropZone
        tier="chunker"
        bounds={bounds}
        isActive={true}
        isHovered={false}
      />
    );

    const rect = container.querySelector('rect');
    expect(rect).toHaveAttribute('x', '100');
    expect(rect).toHaveAttribute('y', '200');
    expect(rect).toHaveAttribute('width', '300');
    expect(rect).toHaveAttribute('height', '100');
  });

  it('centers text in bounds', () => {
    const bounds = { x: 100, y: 200, width: 300, height: 100 };
    const { container } = renderInSVG(
      <TierDropZone
        tier="chunker"
        bounds={bounds}
        isActive={true}
        isHovered={false}
      />
    );

    const text = container.querySelector('text');
    expect(text).toHaveAttribute('x', '250'); // 100 + 300/2
    expect(text).toHaveAttribute('y', '250'); // 200 + 100/2
    expect(text).toHaveAttribute('text-anchor', 'middle');
    expect(text).toHaveAttribute('dominant-baseline', 'middle');
  });

  it('has data-tier attribute', () => {
    const { container } = renderInSVG(
      <TierDropZone
        tier="embedder"
        bounds={defaultBounds}
        isActive={true}
        isHovered={false}
      />
    );

    const group = container.querySelector('g.tier-drop-zone');
    expect(group).toHaveAttribute('data-tier', 'embedder');
  });
});
