CREATE VIEW total_counts AS
  SELECT token, count(token) as count
  FROM sim
  GROUP BY token
  ORDER BY count DESC;

-- % of times a token is >threshold similar to global risk factor mean
-- filter out infrequently occurring tokens
SELECT
       sim.token,
       count(sim.token) as count,
       (count(sim.token)*1.0/(tc.count*1.0)) * 100.0 as percent
FROM sim
INNER JOIN total_counts tc on sim.token = tc.token
WHERE sim.all_similarity > 0.75
and count between 30 and 300
GROUP BY sim.token
ORDER BY percent DESC;

--all_similarity: how similar token is to global mean
select * from sim limit 5;

-- MAIN GOAL:
-- mean("risk factor") for all risk factors (num_risk_factors)
-- for each token
    -- how many times out of num_risk_factors sim("token", mean("risk factor")) > 0.9
    -- e.g. "old", sim("old", mean("diabetes")) == .95 , 1
    -- 11/30


-- for each token (rows - 10M iters)
    -- count = 0
    -- for each similarity to risk factor (cols - ~30 iters)
        -- if similarity > 0.9
            -- count ++
    -- if count > threshold (maybe like 5,10,15 idk - out of 30)
        -- classify as risk factor
select * from
    (
        select token,
               (case when hiv>0.9 then 1 else 0 end) as hiv,
               (case when cell>0.9 then 1 else 0 end) as cell,
               (case when dialysis>0.9 then 1 else 0 end) as dialysis,
               (case when deficient>0.9 then 1 else 0 end) as deficient,
               (case when elderly>0.9 then 1 else 0 end) as elderly,
               (case when nursing>0.9 then 1 else 0 end) as nursing,
               (case when artery>0.9 then 1 else 0 end) as artery,
               (case when liver>0.9 then 1 else 0 end) as liver,
               (case when corticosteroids>0.9 then 1 else 0 end) as corticosteroids,
               (case when smoked>0.9 then 1 else 0 end) as smoked,
               (case when "65">0.9 then 1 else 0 end) as "65",
               (case when cardiomyopathies>0.9 then 1 else 0 end) as cardiomyopathies,
               (case when smoking>0.9 then 1 else 0 end) as smoking,
               (case when bmi>0.9 then 1 else 0 end) as bmi,
               (case when pulmonary>0.9 then 1 else 0 end) as pulmonary,
               (case when cancer>0.9 then 1 else 0 end) as cancer,
               (case when bone>0.9 then 1 else 0 end) as bone,
               (case when sickle>0.9 then 1 else 0 end) as sickle,
               (case when coronary>0.9 then 1 else 0 end) as coronary,
               (case when marrow>0.9 then 1 else 0 end) as marrow,
               (case when hemoglobin>0.9 then 1 else 0 end) as hemoglobin,
               (case when congenital>0.9 then 1 else 0 end) as congenital,
               (case when disease>0.9 then 1 else 0 end) as disease,
               (case when diabetic>0.9 then 1 else 0 end) as diabetic,
               (case when heart>0.9 then 1 else 0 end) as heart,
               (case when obesity>0.9 then 1 else 0 end) as obesity,
               (case when facility>0.9 then 1 else 0 end) as facility
        from sim
        where length(token) > 4
        --where is_hold_out = 'True' -- (best identified risk factors from holdout set: [diabetes, immune, lung]
    )
where hiv + cell + dialysis + deficient + elderly + nursing + artery + liver + corticosteroids
+ smoked + "65" + cardiomyopathies + smoking + bmi + pulmonary + cancer + bone + sickle + coronary
+ marrow + hemoglobin + congenital + disease + diabetic + heart + obesity + facility > 16;
-- mostly just frequently occuring stopwords such as "that", "the", "a", etc.


-------------------
-- HOLDOUT ANALYSIS
-------------------
select token, hiv, is_hold_out
from sim
where hiv > 0.9
and is_hold_out='True';  -- kinda hacky though

-- count of hold out tokens
select is_hold_out, count(*) as count from sim
group by is_hold_out;

-- if hold out tokens are risk factors, they should be more similar to provided risk factors
-- then tokens that are not
-- SWAP out a risk factor in the avg(< >) to check
select avg(artery) from sim
where is_hold_out='True';

-- should be greater than this:

select avg(artery) from sim
where is_hold_out='False';

-- seems to be the case for most risk factors (passes this low baseline)

-- Manual check
-- corticosteroids: .49 > .43
-- sickle: .48 > .45
-- nursing: .51 > .45
-- deficient: .49 > .47
-- heart: .51 > .46
-- smoking: .50 > .44
-- liver: .51 > .45
-- artery: .47 > .46
-- cancer: .50 > .44
-- hiv: .49 > .42
-- average difference: .04 (not very large though)

-- Passes the minimum baseline:
-- avg the similarity of all hold out tokens (risk factors) with the
-- given risk factor hidden state means
-- out of 10 randomly sampled risk factors, all 10 of them had higher hold out averages
-- than the average for non-hold out tokens, indicating that hold out tokens
-- are more similar to the provided risk factors than non-holdout tokens

