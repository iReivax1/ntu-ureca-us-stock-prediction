SELECT COUNT(*) FROM dbmgr_ustwitternewsfeed
GROUP BY posted_by_id;

SELECT * FROM dbmgr_ustwitternewsfeed
where posted_by_id=1
ORDER BY created_datetime;

SELECT mt.*
FROM dbmgr_ustwitternewsfeed mt INNER JOIN
    (
	SELECT min(feedid) as minid
	FROM dbmgr_ustwitternewsfeed
	GROUP BY posted_by_id
    ) t ON mt.feedid = t.minid
ORDER BY mt.posted_by_id;