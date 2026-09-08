import { get } from '@vercel/blob';
import { randomUUID } from 'node:crypto';
import { NextResponse } from 'next/server';
import { reviewer,PREFIX,privateHeaders,sign } from '../../../lib/sigf-auth';
export const runtime='nodejs';
export async function GET(){
 try{
 const existing=await reviewer();
 const code=existing || `R-${randomUUID()}`;
 const blob=await get(`${PREFIX}/manifest.json`,{access:'private',useCache:false});
 if(!blob||blob.statusCode!==200)throw Error('missing');
 const manifest=JSON.parse(await new Response(blob.stream).text());
 const response=NextResponse.json({...manifest,reviewerCode:code},{headers:privateHeaders});
 if(!existing){
  if(!process.env.SIGF_REVIEW_PASSCODE)throw Error('Session secret unavailable');
  const payload=Buffer.from(JSON.stringify({code,exp:Date.now()+30*86400000})).toString('base64url');
  response.cookies.set('sigf-review',`${payload}.${sign(payload)}`,{httpOnly:true,secure:process.env.NODE_ENV==='production',sameSite:'strict',path:'/',maxAge:30*86400});
 }
 return response;
 }catch{return Response.json({error:'The review set is unavailable. Please retry.'},{status:503});}
}
