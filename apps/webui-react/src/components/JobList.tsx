import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useJobsStore } from '../stores/jobsStore';
import { jobsApi } from '../services/api';
import JobCard from './JobCard';

function JobList() {
  const { jobs, setJobs } = useJobsStore();

  const { data, refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: async () => {
      const response = await jobsApi.list();
      return response.data;
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  useEffect(() => {
    if (data) {
      setJobs(data);
    }
  }, [data, setJobs]);

  // Listen for refetch events
  useEffect(() => {
    const handleRefetch = () => {
      refetch();
    };
    window.addEventListener('refetch-jobs', handleRefetch);
    return () => {
      window.removeEventListener('refetch-jobs', handleRefetch);
    };
  }, [refetch]);

  const activeJobs = jobs.filter(
    (job) => job.status === 'processing' || job.status === 'waiting'
  );
  const completedJobs = jobs.filter((job) => job.status === 'completed');
  const failedJobs = jobs.filter((job) => job.status === 'failed');

  // Combine all jobs in order: active first, then completed, then failed
  const sortedJobs = [...activeJobs, ...completedJobs, ...failedJobs];

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Embedding Jobs</h2>
        <button
          onClick={() => refetch()}
          className="inline-flex items-center px-3 py-1.5 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Jobs List */}
      {sortedJobs.length > 0 ? (
        <div className="space-y-4">
          {sortedJobs.map((job) => (
            <JobCard key={job.id} job={job} onDelete={() => refetch()} />
          ))}
        </div>
      ) : (
        /* Empty State */
        <div className="text-center py-12 text-gray-500">
          No jobs found
        </div>
      )}
    </div>
  );
}

export default JobList;