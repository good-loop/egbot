package com.goodloop.egbot.server;

import java.util.Arrays;
import java.util.List;

import com.goodloop.egbot.data.SEQuestion;
import com.winterwell.es.IESRouter;
import com.winterwell.es.client.ESHttpClient;
import com.winterwell.es.client.SearchRequestBuilder;
import com.winterwell.es.client.query.ESQueryBuilder;
import com.winterwell.es.client.query.ESQueryBuilders;
import com.winterwell.es.client.query.MoreLikeThisQueryBuilder;
import com.winterwell.utils.Dep;

public class RelatedQuestionFinder {

	public List<SEQuestion> run(String q) {
		IESRouter router = Dep.get(IESRouter.class);
		ESHttpClient esc = Dep.get(ESHttpClient.class);
		String index = router.getPath(SEQuestion.class, null).index();
		SearchRequestBuilder search = esc.prepareSearch(index);
		search.setDebug(true);
		
		// similarity
		MoreLikeThisQueryBuilder esq = ESQueryBuilders.similar(q, Arrays.asList("body_markdown","egbot_answer_body"));
		esq.setMinTermFreq(1);
		esq.setMinDocFreq(1);
		search.setQuery(esq);
		
		List<SEQuestion> qs = search.get().getSearchResults(SEQuestion.class);
		return qs;
	}

}
