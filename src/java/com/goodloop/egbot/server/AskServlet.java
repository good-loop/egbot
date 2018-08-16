package com.goodloop.egbot.server;

import java.util.ArrayList;
import java.util.List;

import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.web.SimpleJson;
import com.winterwell.web.ajax.JSend;
import com.winterwell.web.ajax.JThing;
import com.winterwell.web.app.IServlet;
import com.winterwell.web.app.WebRequest;

public class AskServlet implements IServlet {

	@Override
	public void process(WebRequest state) throws Exception {
		String q = state.get("q");

		// By Analogy to previous Q
		// Find related Qs
		List relatedQs = new RelatedQuestionFinder().run(q);
		
		// TODO adapt related Q into an answer
		Object answer = "dunno";
		if (relatedQs!=null &&  ! relatedQs.isEmpty()) {
			Object rq0 = relatedQs.get(0);
			answer = SimpleJson.get(rq0, "answers", 0);
		}
		
		ArrayMap data = new ArrayMap(
			"related", relatedQs,
			"answer", answer
				);
		JSend jsend = new JSend(data);
		jsend.send(state);
	}

}
