const backend = process.env.MOTANAXY_BACKEND_URL || 'http://127.0.0.1:8000';
const cloud = process.env.MOTANAXY_CLOUD_BUILD === '1';

export default {
  env: { NEXT_PUBLIC_MOTANAXY_CLOUD: cloud ? '1' : (process.env.NEXT_PUBLIC_MOTANAXY_CLOUD || '0') },
  ...(cloud ? { output: 'export' } : {}),
  ...(cloud ? {} : { async rewrites() {
    return [{ source: '/api/:path*', destination: `${backend}/api/:path*` }];
  } }),
};
