SELECT
    token,
    is_hold_out,
    avg(all_similarity) as average_sim,
    count(*) as total_observed
FROM
     sim
GROUP BY
    token, is_hold_out
ORDER BY
    average_sim DESC;