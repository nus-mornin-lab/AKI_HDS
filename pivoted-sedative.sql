-- This query extracts durations of sedative administration
-- It groups together any administration of the below list of drugs:
-- ItemID DrugName
-- 30118	Fentanyl
-- 30124	Midazolam
-- 30131	Propofol
-- 30149	Fentanyl (Conc)
-- 30150	Fentanyl Base
-- 30163	Dilaudid
-- 30308	Fentanyl Drip

DROP TABLE IF EXISTS pivoted_sedative CASCADE;
CREATE TABLE pivoted_sedative as

with vw0 as
(
  select
    patientunitstayid
    -- due to issue in ETL, times of 0 should likely be null
    , case when drugorderoffset = 0 then null else drugorderoffset end as drugorderoffset
    , case when drugstartoffset = 0 then null else drugstartoffset end as drugstartoffset
    , case when drugstopoffset = 0 then null else drugstopoffset end as drugstopoffset

    -- assign our own identifier based off HICL codes
    -- the following codes have multiple drugs: 35779, 1874, 189
    , case
    	when lower(drugname) like '%fentanyl%' then 'fentanyl' -- Fentanyl, Fentanyl (Conc), Fentanyl Base, Fentanyl Drip
    	when lower(drugname) like '%midazolam%' then 'midazolam' -- Midazolam
    	when lower(drugname) like '%propofol%' then 'propofol' -- Propofol
    	when lower(drugname) like '%dilaudid%' then 'dilaudid' -- Dilaudid
      else null end
        as drugname_structured
    , drugname, drughiclseqno, gtc

    -- delivery info
    , dosage, routeadmin, prn
  from medication m
  -- only non-zero dosages
  where dosage is not null
  -- not cancelled
  and drugordercancelled = 'No'
)
select
    patientunitstayid
  , min(drugstartoffset) as chartoffset
  , max(drugstopoffset) as drugstopoffset
  , max(case 
  			when drugname_structured = 'fentanyl' then 1 
  			when drugname_structured = 'midazolam' then 1 
  			when drugname_structured = 'propofol' then 1 
  			when drugname_structured = 'dilaudid' then 1 
  		else 0 end)::SMALLINT as sedative
from vw0
WHERE
  -- have to have a start time
  drugstartoffset is not null
  -- AND sedative = 1
GROUP BY
  patientunitstayid
ORDER BY
  patientunitstayid;
