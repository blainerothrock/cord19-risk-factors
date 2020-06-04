CREATE VIEW total_counts AS
  SELECT token, count(token) as count
  FROM sim
  GROUP BY token
  ORDER BY count DESC;

SELECT
       sim.token,
       count(sim.token) as count,
       (count(sim.token)*1.0/(tc.count*1.0)) * 100.0 as percent
FROM sim
INNER JOIN total_counts tc on sim.token = tc.token
WHERE sim.all_similarity > 0.9
GROUP BY sim.token
ORDER BY percent DESC;