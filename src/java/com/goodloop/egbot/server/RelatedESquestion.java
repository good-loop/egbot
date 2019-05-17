
package com.goodloop.egbot.server;

import java.util.Arrays;
import java.util.List;

import com.goodloop.egbot.data.ESQuestion;
import com.goodloop.egbot.data.PauliusQuestion;
import com.winterwell.es.IESRouter;
import com.winterwell.es.client.ESHttpClient;
import com.winterwell.es.client.SearchRequestBuilder;
import com.winterwell.es.client.query.ESQueryBuilders;
import com.winterwell.es.client.query.MoreLikeThisQueryBuilder;
import com.winterwell.utils.Dep;

public class RelatedESquestion {

	public List<ESQuestion> run(String q) {
		ESHttpClient esc = Dep.get(ESHttpClient.class);
		SearchRequestBuilder search = esc.prepareSearch(ESModel.indexName).setType(ESModel.indexType);		
		search.setDebug(true);
		
		// similarity
		MoreLikeThisQueryBuilder esq = ESQueryBuilders.similar(q, 
				Arrays.asList("question","answer")); //TODO: do we want most similar to existing q or q+a?
		esq.setMinTermFreq(1);
		esq.setMinDocFreq(1);
		Object hasans = ESQueryBuilders.existsQuery("answers");
		ESQueryBuilders.must(esq, hasans );
		search.setQuery(esq);
		
		List<ESQuestion> qs = search.get().getSearchResults(ESQuestion.class);
		return qs;
	}
}


