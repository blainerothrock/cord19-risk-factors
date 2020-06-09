 SELECT
    (SELECT avg(all_similarity) FROM sim WHERE is_hold_out == 'True') as hold_out_avg,
    (SELECT avg(all_similarity) FROM sim WHERE is_hold_out == 'False') as non_hold_out_avg