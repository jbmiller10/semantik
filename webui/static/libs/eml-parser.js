// Simple EML parser for email viewing
window.emlFormat = {
    parse: function(emlText) {
        const lines = emlText.split('\n');
        const headers = {};
        let bodyStartIndex = -1;
        
        // Parse headers
        let currentHeader = null;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // Empty line marks end of headers
            if (line.trim() === '') {
                bodyStartIndex = i + 1;
                break;
            }
            
            // Continuation of previous header
            if (line.startsWith(' ') || line.startsWith('\t')) {
                if (currentHeader) {
                    headers[currentHeader] += ' ' + line.trim();
                }
            } else {
                // New header
                const colonIndex = line.indexOf(':');
                if (colonIndex > 0) {
                    currentHeader = line.substring(0, colonIndex).toLowerCase();
                    headers[currentHeader] = line.substring(colonIndex + 1).trim();
                }
            }
        }
        
        // Extract body
        let body = '';
        let html = '';
        if (bodyStartIndex >= 0) {
            const bodyLines = lines.slice(bodyStartIndex);
            
            // Check for multipart messages
            const contentType = headers['content-type'] || '';
            if (contentType.includes('multipart')) {
                // Simple multipart parsing
                const boundary = contentType.match(/boundary=["']?([^"';]+)/i);
                if (boundary) {
                    const parts = emlText.split('--' + boundary[1]);
                    for (const part of parts) {
                        if (part.includes('Content-Type: text/plain')) {
                            const plainStart = part.indexOf('\n\n');
                            if (plainStart >= 0) {
                                body = part.substring(plainStart + 2).replace(/\r\n/g, '\n').trim();
                            }
                        } else if (part.includes('Content-Type: text/html')) {
                            const htmlStart = part.indexOf('\n\n');
                            if (htmlStart >= 0) {
                                html = part.substring(htmlStart + 2).replace(/\r\n/g, '\n').trim();
                            }
                        }
                    }
                }
            } else {
                body = bodyLines.join('\n').trim();
            }
        }
        
        // Clean up encoded content
        if (headers['content-transfer-encoding'] === 'quoted-printable') {
            body = body.replace(/=\r?\n/g, '').replace(/=([0-9A-F]{2})/gi, 
                (match, hex) => String.fromCharCode(parseInt(hex, 16)));
            html = html.replace(/=\r?\n/g, '').replace(/=([0-9A-F]{2})/gi, 
                (match, hex) => String.fromCharCode(parseInt(hex, 16)));
        }
        
        return {
            headers: headers,
            subject: headers['subject'] || 'No Subject',
            from: {
                text: headers['from'] || ''
            },
            to: {
                text: headers['to'] || ''
            },
            date: headers['date'] || '',
            text: body,
            html: html || null
        };
    }
};