import AdminSettings from './AdminSettings';

/**
 * SystemTab displays read-only system information and health status.
 * This tab is visible to all users and wraps the existing AdminSettings component.
 */
export default function SystemTab() {
  return <AdminSettings />;
}
