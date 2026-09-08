import {get,list} from '@vercel/blob';
import {PREFIX,equal,privateHeaders} from '../../../../lib/sigf-auth';
import {finalRating,type StoredRating} from '../../../../lib/sigf-review';
export const runtime='nodejs';
function cell(v:unknown){let s=String(v??'');if(/^[=+@\-\t\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';}
export async function GET(req:Request){
 const key=req.headers.get('x-admin-key')||'',expected=process.env.SIGF_ADMIN_EXPORT_TOKEN || process.env.ADMIN_EXPORT_TOKEN;
 if(!expected||!equal(key,expected))return Response.json({error:'Check the results passcode.'},{status:401});
 try{const blobs=[];let cursor:string|undefined;do{const p=await list({prefix:`${PREFIX}/responses/`,limit:1000,cursor});blobs.push(...p.blobs);cursor=p.hasMore?p.cursor:undefined;}while(cursor);
 const records=await Promise.all(blobs.map(async b=>{const r=await get(b.pathname,{access:'private',useCache:false});return r&&r.statusCode===200?JSON.parse(await new Response(r.stream).text()):null;}));
 const cols=['studyVersion','reviewerCode','caseId','followup','years','similarity','observed','change','notes','savedAt'];const lines=[cols.join(',')];
 for(const r of records.filter(Boolean)){if(r.reviewerCode.startsWith('SYSTEM-TEST'))continue;const a=finalRating<StoredRating>(r.ratings);if(!a)continue;const row={...r,...a,followup:a.index+1};lines.push(cols.map(k=>cell(row[k])).join(','));}
 return new Response(lines.join('\r\n'),{headers:{...privateHeaders,'Content-Type':'text/csv; charset=utf-8','Content-Disposition':'attachment; filename="SIGF-advisor-ratings.csv"'}});
 }catch{return Response.json({error:'Export unavailable. Please retry.'},{status:503});}
}
