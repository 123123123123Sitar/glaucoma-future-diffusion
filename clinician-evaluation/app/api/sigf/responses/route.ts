import { get,put,list } from '@vercel/blob';
import { reviewer,PREFIX,VERSION,privateHeaders } from '../../../../lib/sigf-auth';
import {parseFinalSubmission,mergeFinalRating} from '../../../../lib/sigf-review';
export const runtime='nodejs';
export async function GET(){
 const code=await reviewer();if(!code)return Response.json({error:'Open the review page to start.'},{status:401});
 try{const page=await list({prefix:`${PREFIX}/responses/${code}/`,limit:1000});
 const records=await Promise.all(page.blobs.map(async b=>{const r=await get(b.pathname,{access:'private',useCache:false});return r&&r.statusCode===200?JSON.parse(await new Response(r.stream).text()):null;}));
 return Response.json({records:records.filter(Boolean)},{headers:privateHeaders});
 }catch{return Response.json({error:'Saved responses could not be loaded.'},{status:503});}
}
export async function POST(req:Request){
 const code=await reviewer();if(!code)return Response.json({error:'Open the review page to start.'},{status:401});
 try{const body=await req.json();
 if(!/^S(0[1-9]|[12][0-9]|3[0-9]|40)$/.test(body.caseId))return Response.json({error:'Invalid case.'},{status:400});
 const submitted=parseFinalSubmission(body.ratings);
 if(!submitted)return Response.json({error:'Please answer the questions for the final follow-up only. Refresh the page if you still see the older grading form.'},{status:400});
 const m=await get(`${PREFIX}/manifest.json`,{access:'private',useCache:false});if(!m||m.statusCode!==200)throw Error('manifest');
 const manifest=JSON.parse(await new Response(m.stream).text());const item=manifest.cases.find((x:any)=>x.id===body.caseId);if(!item)throw Error('case');
 const pathname=`${PREFIX}/responses/${code}/${body.caseId}.json`;
 const previous=await get(pathname,{access:'private',useCache:false});
 if(previous&&previous.statusCode!==200)throw Error('Previous answers unavailable');
 const existing=previous?JSON.parse(await new Response(previous.stream).text()):null;
 const ratings=mergeFinalRating(Array.isArray(existing?.ratings)?existing.ratings:[],submitted,item.years);
 const record={studyVersion:VERSION,reviewScope:'final-followup',reviewerCode:code,caseId:body.caseId,ratings,savedAt:new Date().toISOString()};
 await put(pathname,JSON.stringify(record),{access:'private',contentType:'application/json',addRandomSuffix:false,allowOverwrite:true});
 return Response.json({saved:true,record},{headers:privateHeaders});
 }catch{return Response.json({error:'Your answers were not saved. Please retry.'},{status:500});}
}
