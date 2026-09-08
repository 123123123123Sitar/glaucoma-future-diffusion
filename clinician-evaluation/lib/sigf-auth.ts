import { createHmac, timingSafeEqual } from 'node:crypto';
import { cookies } from 'next/headers';
export const VERSION = 'sigf-advisor-curated-20260904-v1';
export const PREFIX = `sigf/${VERSION}`;
export function equal(a: string, b: string) { const x=Buffer.from(a),y=Buffer.from(b);return x.length===y.length && timingSafeEqual(x,y); }
export function sign(value: string) { return createHmac('sha256',process.env.SIGF_REVIEW_PASSCODE || 'disabled').update(value).digest('hex'); }
export async function reviewer() {
 if (!process.env.SIGF_REVIEW_PASSCODE) return null;
 const raw=(await cookies()).get('sigf-review')?.value || '';const [payload,signature]=raw.split('.');
 if (!payload || !signature || !equal(sign(payload),signature)) return null;
 try {const p=JSON.parse(Buffer.from(payload,'base64url').toString());return /^[A-Za-z0-9-]{2,40}$/.test(p.code)&&p.exp>Date.now()?p.code as string:null;}catch{return null;}
}
export const privateHeaders={'Cache-Control':'private, no-store','X-Robots-Tag':'noindex, nofollow'};
