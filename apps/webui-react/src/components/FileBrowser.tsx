import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Folder, File, ChevronRight, ArrowUp, Loader2, HardDrive } from 'lucide-react';
import apiClient from '../services/api/v2/client';

interface FileItem {
    name: string;
    type: 'dir' | 'file';
    path: string;
    size?: number;
}

interface FileListResponse {
    items: FileItem[];
    current_path: string;
    parent_path: string | null;
}

interface FileBrowserProps {
    onSelect: (path: string) => void;
    initialPath?: string;
    className?: string;
}

export function FileBrowser({ onSelect, initialPath, className = '' }: FileBrowserProps) {
    const [currentPath, setCurrentPath] = useState<string | undefined>(initialPath);

    const { data, isLoading, error, refetch } = useQuery<FileListResponse>({
        queryKey: ['fs', 'list', currentPath],
        queryFn: async () => {
            const params = new URLSearchParams();
            if (currentPath) params.append('path', currentPath);
            const res = await apiClient.get(`/fs/list?${params.toString()}`);
            return res.data;
        },
    });

    const handleNavigate = (path: string) => {
        // If it's an absolute path from the API, we might need to convert it to relative 
        // for the next request if the API expects relative paths for navigation.
        // However, our API implementation handles absolute paths if they are within root,
        // or relative paths. The API returns absolute paths in 'path' field.
        // Let's try sending the absolute path.

        // Check if we are navigating up
        if (data && path === data.parent_path) {
            // Going up
        }

        setCurrentPath(path);
        onSelect(path);
    };

    const handleUp = () => {
        if (data?.parent_path !== undefined) {
            // If parent_path is empty string, it means root.
            // If it's null, we are at root (or can't go up).
            if (data.parent_path === null) return;
            setCurrentPath(data.parent_path);
        }
    };

    const formatSize = (bytes?: number) => {
        if (bytes === undefined) return '';
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    };

    if (error) {
        return (
            <div className={`p-4 border border-red-200 rounded bg-red-50 text-red-600 ${className}`}>
                <p>Error loading directory listing.</p>
                <button
                    onClick={() => refetch()}
                    className="mt-2 text-sm underline hover:text-red-800"
                >
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className={`border rounded-lg overflow-hidden bg-white shadow-sm ${className}`}>
            {/* Header / Breadcrumbs */}
            <div className="bg-gray-50 p-3 border-b flex items-center gap-2">
                <button
                    onClick={handleUp}
                    disabled={!data?.parent_path && data?.parent_path !== ""}
                    className="p-1 hover:bg-gray-200 rounded disabled:opacity-30 disabled:cursor-not-allowed"
                    title="Go Up"
                >
                    <ArrowUp size={18} />
                </button>

                <div className="flex-1 font-mono text-sm truncate flex items-center text-gray-600">
                    <HardDrive size={16} className="mr-2 text-gray-400" />
                    {data?.current_path || 'Loading...'}
                </div>

                {isLoading && <Loader2 size={18} className="animate-spin text-blue-500" />}
            </div>

            {/* File List */}
            <div className="max-h-60 overflow-y-auto">
                {isLoading && !data ? (
                    <div className="p-8 text-center text-gray-400">Loading files...</div>
                ) : (
                    <ul className="divide-y divide-gray-100">
                        {data?.items.length === 0 && (
                            <li className="p-4 text-center text-gray-400 italic">Empty directory</li>
                        )}
                        {data?.items.map((item) => (
                            <li
                                key={item.name}
                                onClick={() => item.type === 'dir' ? handleNavigate(item.path) : onSelect(item.path)}
                                className={`
                    flex items-center gap-3 p-2 hover:bg-blue-50 cursor-pointer transition-colors
                    ${item.type === 'dir' ? 'text-gray-700' : 'text-gray-500'}
                `}
                            >
                                {item.type === 'dir' ? (
                                    <Folder size={18} className="text-blue-400 fill-blue-100" />
                                ) : (
                                    <File size={18} className="text-gray-400" />
                                )}
                                <span className="flex-1 truncate text-sm">{item.name}</span>
                                {item.size !== undefined && (
                                    <span className="text-xs text-gray-400 font-mono">{formatSize(item.size)}</span>
                                )}
                                {item.type === 'dir' && <ChevronRight size={14} className="text-gray-300" />}
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {/* Footer / Selection Status */}
            <div className="bg-gray-50 p-2 text-xs text-gray-500 border-t flex justify-between">
                <span>{data?.items.filter(i => i.type === 'dir').length || 0} folders</span>
                <span>{data?.items.filter(i => i.type === 'file').length || 0} files</span>
            </div>
        </div>
    );
}
