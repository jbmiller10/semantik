<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api/client';
	import { auth } from '$lib/stores/auth';

	interface SystemStats {
		total_jobs: number;
		total_documents: number;
		total_vectors: number;
		disk_usage: {
			embeddings_size_mb: number;
			database_size_mb: number;
		};
	}

	let stats: SystemStats | null = null;
	let loading = true;
	let resetting = false;

	onMount(async () => {
		await loadStats();
	});

	async function loadStats() {
		try {
			stats = await api.get<SystemStats>('/api/settings/stats');
		} catch (error) {
			console.error('Failed to load stats:', error);
		} finally {
			loading = false;
		}
	}

	async function resetDatabase() {
		if (!confirm('Are you sure you want to reset the database? This will delete ALL jobs, documents, and vectors. This action cannot be undone!')) {
			return;
		}

		if (!confirm('This is your last chance to cancel. All data will be permanently deleted. Continue?')) {
			return;
		}

		resetting = true;
		try {
			await api.post('/api/settings/reset-database');
			alert('Database reset successfully');
			await loadStats();
		} catch (error) {
			alert('Failed to reset database');
		} finally {
			resetting = false;
		}
	}
</script>

<div class="container mx-auto px-4 py-8">
	<!-- Header -->
	<div class="flex justify-between items-center mb-8">
		<h1 class="text-3xl font-bold text-gray-800">
			<i class="fas fa-cog mr-2"></i>System Settings
		</h1>
		<a href="/" class="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors">
			<i class="fas fa-arrow-left mr-2"></i>Back to Dashboard
		</a>
	</div>

	<!-- User Info -->
	<div class="bg-white rounded-lg shadow-md p-6 mb-6">
		<h2 class="text-xl font-semibold mb-4">User Information</h2>
		<div class="grid grid-cols-2 gap-4">
			<div>
				<span class="text-gray-600">Username:</span>
				<span class="font-medium ml-2">{$auth.user?.username}</span>
			</div>
			<div>
				<span class="text-gray-600">Email:</span>
				<span class="font-medium ml-2">{$auth.user?.email}</span>
			</div>
			{#if $auth.user?.full_name}
				<div class="col-span-2">
					<span class="text-gray-600">Full Name:</span>
					<span class="font-medium ml-2">{$auth.user.full_name}</span>
				</div>
			{/if}
		</div>
	</div>

	<!-- System Statistics -->
	{#if loading}
		<div class="bg-white rounded-lg shadow-md p-6">
			<div class="animate-pulse">
				<div class="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
				<div class="space-y-3">
					<div class="h-4 bg-gray-200 rounded w-3/4"></div>
					<div class="h-4 bg-gray-200 rounded w-1/2"></div>
					<div class="h-4 bg-gray-200 rounded w-2/3"></div>
				</div>
			</div>
		</div>
	{:else if stats}
		<div class="bg-white rounded-lg shadow-md p-6 mb-6">
			<h2 class="text-xl font-semibold mb-4">System Statistics</h2>
			<div class="grid grid-cols-2 md:grid-cols-4 gap-6">
				<div class="text-center">
					<div class="text-3xl font-bold text-blue-600">{stats.total_jobs}</div>
					<div class="text-gray-600">Total Jobs</div>
				</div>
				<div class="text-center">
					<div class="text-3xl font-bold text-green-600">{stats.total_documents.toLocaleString()}</div>
					<div class="text-gray-600">Documents</div>
				</div>
				<div class="text-center">
					<div class="text-3xl font-bold text-purple-600">{stats.total_vectors.toLocaleString()}</div>
					<div class="text-gray-600">Vectors</div>
				</div>
				<div class="text-center">
					<div class="text-3xl font-bold text-orange-600">
						{(stats.disk_usage.embeddings_size_mb + stats.disk_usage.database_size_mb).toFixed(1)} MB
					</div>
					<div class="text-gray-600">Disk Usage</div>
				</div>
			</div>
			
			<div class="mt-6 pt-6 border-t">
				<h3 class="font-medium text-gray-700 mb-2">Storage Breakdown</h3>
				<div class="space-y-2 text-sm">
					<div class="flex justify-between">
						<span class="text-gray-600">Embeddings:</span>
						<span class="font-medium">{stats.disk_usage.embeddings_size_mb.toFixed(1)} MB</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-600">Database:</span>
						<span class="font-medium">{stats.disk_usage.database_size_mb.toFixed(1)} MB</span>
					</div>
				</div>
			</div>
		</div>
	{/if}

	<!-- Danger Zone -->
	<div class="bg-red-50 border-2 border-red-200 rounded-lg p-6">
		<h2 class="text-xl font-semibold text-red-800 mb-4">
			<i class="fas fa-exclamation-triangle mr-2"></i>Danger Zone
		</h2>
		<p class="text-red-700 mb-4">
			The following actions are destructive and cannot be undone. Please proceed with caution.
		</p>
		<button
			on:click={resetDatabase}
			disabled={resetting}
			class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
		>
			{#if resetting}
				<i class="fas fa-spinner fa-spin mr-2"></i>Resetting...
			{:else}
				<i class="fas fa-database mr-2"></i>Reset Database
			{/if}
		</button>
	</div>
</div>