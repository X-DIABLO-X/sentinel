/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The console talks to two FastAPI processes over CORS (both enable
  // allow_origins=["*"]), so no rewrites/proxy are needed. Base URLs come
  // from NEXT_PUBLIC_CCTV_API / NEXT_PUBLIC_DRONE_API - see lib/api.ts.
  eslint: { ignoreDuringBuilds: true },
};

module.exports = nextConfig;
