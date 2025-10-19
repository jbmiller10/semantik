type NavigationHandler = (path: string) => void

let navigateHandler: NavigationHandler | null = null

export function registerNavigationHandler(handler: NavigationHandler) {
  navigateHandler = handler
}

export function navigateTo(path: string) {
  if (navigateHandler) {
    navigateHandler(path)
    return
  }

  if (typeof window !== 'undefined') {
    window.location.assign(path)
  }
}
