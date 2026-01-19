import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect } from 'vitest';
import { useState } from 'react';
import { ConfigMatrixBuilder } from '../ConfigMatrixBuilder';
import type { ConfigMatrixItem } from '../../../types/benchmark';

function Wrapper(props: { initial: ConfigMatrixItem; hasSparseIndex?: boolean; hasReranker?: boolean }) {
  const [value, setValue] = useState<ConfigMatrixItem>(props.initial);
  return (
    <ConfigMatrixBuilder
      value={value}
      onChange={setValue}
      hasSparseIndex={props.hasSparseIndex}
      hasReranker={props.hasReranker}
    />
  );
}

describe('ConfigMatrixBuilder', () => {
  it('shows config count and warning when many configurations selected', async () => {
    const user = userEvent.setup();

    render(
      <Wrapper
        hasSparseIndex={true}
        hasReranker={true}
        initial={{
          search_modes: ['dense'],
          use_reranker: [false],
          top_k_values: [10],
          rrf_k_values: [60],
          score_thresholds: [null],
        }}
      />
    );

    expect(screen.getByText('1 configuration selected')).toBeInTheDocument();
    expect(screen.getByText(/Each configuration will be evaluated against all queries/i)).toBeInTheDocument();

    const searchModesBlock = screen.getByText('Search Modes').closest('div');
    expect(searchModesBlock).toBeTruthy();
    const searchModes = within(searchModesBlock as HTMLElement);

    // Select all search modes (dense already selected)
    await user.click(searchModes.getByRole('button', { name: /^Sparse/i }));
    await user.click(searchModes.getByRole('button', { name: /^Hybrid/i }));

    // Include reranker on/off
    await user.click(screen.getByRole('button', { name: /With Reranker/i }));

    // Select all Top-K options (10 already selected)
    const topKBlock = screen.getByText('Top-K Values').closest('div');
    expect(topKBlock).toBeTruthy();
    const topK = within(topKBlock as HTMLElement);

    await user.click(topK.getByRole('button', { name: '5' }));
    await user.click(topK.getByRole('button', { name: '20' }));
    await user.click(topK.getByRole('button', { name: '50' }));
    await user.click(topK.getByRole('button', { name: '100' }));

    // 3 modes * 2 reranker options * 5 top-k values = 30
    expect(screen.getByText('30 configurations selected')).toBeInTheDocument();
    expect(screen.getByText(/Large number of configurations may take a while to run/i)).toBeInTheDocument();
  });

  it('disables sparse and hybrid when hasSparseIndex is false', () => {
    render(
      <Wrapper
        hasSparseIndex={false}
        initial={{
          search_modes: ['dense'],
          use_reranker: [false],
          top_k_values: [10],
          rrf_k_values: [60],
          score_thresholds: [null],
        }}
      />
    );

    const searchModesBlock = screen.getByText('Search Modes').closest('div');
    expect(searchModesBlock).toBeTruthy();
    const searchModes = within(searchModesBlock as HTMLElement);

    const sparse = searchModes.getByRole('button', { name: /^Sparse/i });
    const hybrid = searchModes.getByRole('button', { name: /^Hybrid/i });
    expect(sparse).toBeDisabled();
    expect(hybrid).toBeDisabled();
    expect(screen.getAllByText(/\(no sparse index\)/i).length).toBeGreaterThanOrEqual(1);
  });
});
