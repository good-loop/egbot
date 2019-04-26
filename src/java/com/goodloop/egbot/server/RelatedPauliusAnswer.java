package com.goodloop.egbot.server;

import java.util.Arrays;
import java.util.List;

import com.goodloop.egbot.data.PauliusQuestion;
import com.winterwell.es.IESRouter;
import com.winterwell.es.client.ESHttpClient;
import com.winterwell.es.client.SearchRequestBuilder;
import com.winterwell.es.client.query.ESQueryBuilders;
import com.winterwell.es.client.query.MoreLikeThisQueryBuilder;
import com.winterwell.utils.Dep;

public class RelatedPauliusAnswer {

	public List<PauliusQuestion> run(String q) {
		IESRouter router = Dep.get(IESRouter.class);
		ESHttpClient esc = Dep.get(ESHttpClient.class);
		String index = router.getPath(PauliusQuestion.class, null).index();
		SearchRequestBuilder search = esc.prepareSearch(index);
		search.setDebug(true);
		
		// similarity
		MoreLikeThisQueryBuilder esq = ESQueryBuilders.similar(q, Arrays.asList("question","answer"));
		esq.setMinTermFreq(1);
		esq.setMinDocFreq(1);
		Object hasans = ESQueryBuilders.existsQuery("answers");
		ESQueryBuilders.must(esq, hasans );
		search.setQuery(esq);
		
		List<PauliusQuestion> qs = search.get().getSearchResults(PauliusQuestion.class);
		return qs;
	}

}
