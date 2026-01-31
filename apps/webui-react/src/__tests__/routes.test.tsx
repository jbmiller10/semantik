import { describe, it, expect } from 'vitest';
import { render, screen } from '../tests/utils/test-utils';
import App from '../App';

describe('Routes', () => {
  it('does not have /pipeline route', () => {
    render(<App />, { initialEntries: ['/pipeline/test-id'] });

    // Pipeline route should not match - should show 404 or redirect
    // Since PipelineBuilderPage doesn't exist, it shouldn't render pipeline content
    expect(screen.queryByText(/loading pipeline builder/i)).not.toBeInTheDocument();
  });
});
