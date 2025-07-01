<script lang="ts">
	import { auth } from '$lib/stores/auth';
	import Tabs from '$lib/components/ui/Tabs.svelte';
	import CreateJobForm from '$lib/components/CreateJobForm.svelte';
	import JobList from '$lib/components/JobList.svelte';
	import SearchInterface from '$lib/components/SearchInterface.svelte';

	const tabs = [
		{ id: 'create', label: 'Create Embeddings', icon: 'fas fa-plus-circle' },
		{ id: 'jobs', label: 'Jobs', icon: 'fas fa-tasks' },
		{ id: 'search', label: 'Search', icon: 'fas fa-search' }
	];

	let activeTab = 'create';
</script>

<div class="container mx-auto px-4 py-8">
	<!-- Header -->
	<div class="flex justify-between items-center mb-8">
		<h1 class="text-3xl font-bold text-gray-800">
			<i class="fas fa-database mr-2"></i>Document Embedding System
		</h1>
		<div class="flex items-center space-x-4">
			<span class="text-gray-600">
				<i class="fas fa-user mr-2"></i>
				<span>{$auth.user?.username || 'Loading...'}</span>
			</span>
			<a href="/settings" class="text-gray-600 hover:text-gray-800">
				<i class="fas fa-cog text-2xl"></i>
			</a>
			<button
				on:click={() => auth.logout()}
				class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
			>
				<i class="fas fa-sign-out-alt mr-2"></i>Logout
			</button>
		</div>
	</div>

	<!-- Tabs -->
	<div class="mb-8">
		<Tabs {tabs} bind:activeTab />
	</div>

	<!-- Tab Content -->
	<div class="tab-content">
		{#if activeTab === 'create'}
			<CreateJobForm />
		{:else if activeTab === 'jobs'}
			<JobList />
		{:else if activeTab === 'search'}
			<SearchInterface />
		{/if}
	</div>
</div>