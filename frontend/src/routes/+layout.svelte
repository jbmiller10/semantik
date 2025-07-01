<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { auth } from '$lib/stores/auth';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';

	let loading = true;

	onMount(async () => {
		// Skip auth check on login page
		if ($page.url.pathname === '/login') {
			loading = false;
			return;
		}

		const isValid = await auth.checkAuth();
		if (!isValid) {
			goto('/login');
		}
		loading = false;
	});
</script>

{#if loading}
	<div class="flex items-center justify-center min-h-screen">
		<div class="text-gray-600">Loading...</div>
	</div>
{:else}
	<slot />
{/if}