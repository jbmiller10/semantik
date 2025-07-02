import { useState } from 'react';
import { 
  featureChecklist, 
  getImplementationStats, 
  getMissingFeatures,
  type FeatureCheck 
} from '../utils/featureChecklist';

function FeatureVerification() {
  const [checklist, setChecklist] = useState(featureChecklist);
  const [showOnlyUntested, setShowOnlyUntested] = useState(false);

  const stats = getImplementationStats();
  const missingFeatures = getMissingFeatures();

  const toggleTested = (index: number) => {
    const updated = [...checklist];
    updated[index].tested = !updated[index].tested;
    setChecklist(updated);
  };

  const filteredChecklist = showOnlyUntested 
    ? checklist.filter(f => f.implemented && !f.tested)
    : checklist;

  const groupedFeatures = filteredChecklist.reduce((acc, feature, index) => {
    if (!acc[feature.category]) {
      acc[feature.category] = [];
    }
    acc[feature.category].push({ ...feature, originalIndex: index });
    return acc;
  }, {} as Record<string, (FeatureCheck & { originalIndex: number })[]>);

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-6">Feature Verification Checklist</h2>
      
      {/* Stats */}
      <div className="mb-6 grid grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded">
          <div className="text-sm text-blue-600 font-medium">Total Features</div>
          <div className="text-2xl font-bold text-blue-900">{stats.total}</div>
        </div>
        <div className="bg-green-50 p-4 rounded">
          <div className="text-sm text-green-600 font-medium">Implemented</div>
          <div className="text-2xl font-bold text-green-900">
            {stats.implemented} ({stats.implementationPercentage}%)
          </div>
        </div>
        <div className="bg-purple-50 p-4 rounded">
          <div className="text-sm text-purple-600 font-medium">Tested</div>
          <div className="text-2xl font-bold text-purple-900">
            {stats.tested} ({stats.testingPercentage}%)
          </div>
        </div>
      </div>

      {/* Missing Features Warning */}
      {missingFeatures.length > 0 && (
        <div className="mb-6 bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <h3 className="text-sm font-medium text-yellow-800 mb-2">
            Missing Features ({missingFeatures.length})
          </h3>
          <ul className="text-sm text-yellow-700 list-disc list-inside">
            {missingFeatures.map((feature, i) => (
              <li key={i}>
                {feature.feature}
                {feature.notes && <span className="text-yellow-600"> - {feature.notes}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Filter */}
      <div className="mb-4">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={showOnlyUntested}
            onChange={(e) => setShowOnlyUntested(e.target.checked)}
            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <span className="ml-2 text-sm text-gray-700">
            Show only untested features
          </span>
        </label>
      </div>

      {/* Feature List */}
      <div className="space-y-6">
        {Object.entries(groupedFeatures).map(([category, features]) => (
          <div key={category}>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">{category}</h3>
            <div className="space-y-2">
              {features.map((feature) => (
                <div
                  key={feature.originalIndex}
                  className={`flex items-start p-3 rounded-md ${
                    !feature.implemented
                      ? 'bg-gray-50 opacity-60'
                      : feature.tested
                      ? 'bg-green-50'
                      : 'bg-yellow-50'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={feature.tested}
                    onChange={() => toggleTested(feature.originalIndex)}
                    disabled={!feature.implemented}
                    className="mt-0.5 rounded border-gray-300 text-green-600 focus:ring-green-500 disabled:opacity-50"
                  />
                  <div className="ml-3 flex-1">
                    <div className="text-sm font-medium text-gray-900">
                      {feature.feature}
                    </div>
                    {feature.notes && (
                      <div className="text-xs text-gray-600 mt-1">{feature.notes}</div>
                    )}
                  </div>
                  <div className="ml-3">
                    {!feature.implemented ? (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                        Not Implemented
                      </span>
                    ) : feature.tested ? (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                        Tested
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                        Pending Test
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Export Results */}
      <div className="mt-8 flex justify-end">
        <button
          onClick={() => {
            const results = {
              stats,
              checklist: checklist.map(f => ({
                ...f,
                status: !f.implemented ? 'not_implemented' : f.tested ? 'tested' : 'untested'
              })),
              timestamp: new Date().toISOString()
            };
            const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `feature-verification-${new Date().toISOString().split('T')[0]}.json`;
            a.click();
          }}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          Export Results
        </button>
      </div>
    </div>
  );
}

export default FeatureVerification;