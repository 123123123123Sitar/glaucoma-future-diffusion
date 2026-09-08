import { get } from '@vercel/blob';
import { reviewer,PREFIX,privateHeaders } from '../../../../lib/sigf-auth';
export const runtime='nodejs';
export async function GET(req:Request){
 if(!await reviewer())return new Response('Open the review page to start.',{status:401});
 const url=new URL(req.url),id=url.searchParams.get('case')||'',name=url.searchParams.get('name')||'';
 if(!/^S(0[1-9]|[12][0-9]|3[0-9]|40)$/.test(id)||!/^(baseline|real-[012]|ai-[012])$/.test(name))return new Response('Invalid image.',{status:400});
 const blob=await get(`${PREFIX}/images/${id}/${name}.png`,{access:'private'});
 if(!blob||blob.statusCode!==200)return new Response('Image unavailable.',{status:404});
 return new Response(blob.stream,{headers:{...privateHeaders,'Content-Type':'image/png','X-Content-Type-Options':'nosniff'}});
}
