# RetinaProgress clinician study

Direct-link, validation-only blinded review of real and diffusion-generated
longitudinal fundus images.

- Opening `/` starts case 1; there is no landing page.
- The public manifest contains opaque candidate A/B paths and no answer key.
- Case order is randomized per session.
- Ratings are saved centrally after every case through the study API.
- Cloudflare D1 is the authoritative database on Sites.
- The Vercel build can forward writes by setting `SITES_API_URL` to the Sites
  deployment.
- Locked-test images are not included.
- Deployment validation writes one row with reviewer code `SYSTEM-TEST`; exclude
  that row from study analyses.

This is a formative research pilot, not a clinical or diagnostic system.

```bash
npm test
npx next build
```
