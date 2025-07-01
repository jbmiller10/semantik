<script lang="ts">
	import { auth } from '$lib/stores/auth';

	let username = '';
	let password = '';
	let error = '';
	let loading = false;

	async function handleLogin() {
		error = '';
		loading = true;

		try {
			await auth.login(username, password);
		} catch (e) {
			error = 'Invalid username or password';
		} finally {
			loading = false;
		}
	}
</script>

<div class="min-h-screen bg-gray-100 flex items-center justify-center">
	<div class="bg-white p-8 rounded-lg shadow-md w-96">
		<h1 class="text-2xl font-bold text-center mb-8">
			<i class="fas fa-database mr-2"></i>Document Embedding System
		</h1>

		<form on:submit|preventDefault={handleLogin} class="space-y-4">
			<div>
				<label for="username" class="block text-sm font-medium text-gray-700 mb-1">
					Username
				</label>
				<input
					type="text"
					id="username"
					bind:value={username}
					required
					class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
					placeholder="Enter your username"
				/>
			</div>

			<div>
				<label for="password" class="block text-sm font-medium text-gray-700 mb-1">
					Password
				</label>
				<input
					type="password"
					id="password"
					bind:value={password}
					required
					class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
					placeholder="Enter your password"
				/>
			</div>

			{#if error}
				<div class="text-red-600 text-sm text-center">{error}</div>
			{/if}

			<button
				type="submit"
				disabled={loading}
				class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
			>
				{#if loading}
					<i class="fas fa-spinner fa-spin mr-2"></i>
					Logging in...
				{:else}
					<i class="fas fa-sign-in-alt mr-2"></i>
					Login
				{/if}
			</button>
		</form>
	</div>
</div>