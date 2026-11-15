export function remainingEta(job,now=Date.now()){
  if(job?.state!=='running'||job.cancel_requested||job.stage==='cleanup'||!Number.isFinite(job.eta_seconds))return null;
  const updated=Date.parse(job.eta_updated_at);
  const elapsed=Number.isFinite(updated)?Math.max(0,(now-updated)/1000):0;
  return Math.max(0,job.eta_seconds-elapsed);
}
export function formatEta(seconds){
  const value=Math.max(0,Math.ceil(seconds)),hours=Math.floor(value/3600),minutes=Math.floor(value%3600/60);
  return hours?`${hours}h ${minutes}m`:minutes?`${minutes}m ${value%60}s`:`${value}s`;
}
