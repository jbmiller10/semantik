import type { ReactElement } from 'react'
import { render, type RenderOptions } from '@testing-library/react'
import { AllTheProviders } from './providers'

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialEntries?: string[]
}

// Custom render function that includes all providers
const customRender = (
  ui: ReactElement,
  options?: CustomRenderOptions
) => {
  const { initialEntries, ...renderOptions } = options || {}
  
  return render(ui, { 
    wrapper: ({ children }) => (
      <AllTheProviders initialEntries={initialEntries}>
        {children}
      </AllTheProviders>
    ),
    ...renderOptions 
  })
}

// Re-export everything from React Testing Library
// eslint-disable-next-line react-refresh/only-export-components
export * from '@testing-library/react'

// Override the default render with our custom render
export { customRender as render }

// Re-export utilities from queryClient
export { createTestQueryClient } from './queryClient'