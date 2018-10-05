package com.goodloop.egbot.server;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

import jep.Jep;
import jep.JepConfig;
import jep.JepException;

import com.winterwell.es.fail.ESException;
import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.log.Log;
import com.winterwell.utils.web.SimpleJson;
import com.winterwell.web.ajax.JSend;
import com.winterwell.web.ajax.JThing;
import com.winterwell.web.app.IServlet;
import com.winterwell.web.app.WebRequest;

public class AskServlet implements IServlet {

	@Override
	public void process(WebRequest state) throws Exception {
		String q = state.get("q");

		Object answer;
		List relatedQs = findRelatedQuestion(q);
		List relatedAs = findRelatedAnswer(relatedQs);
		Object generatedAnswer = generateAnswer(q);
		
		ArrayMap data = new ArrayMap(
			"relatedQs", relatedQs,
			"relatedAs", relatedAs,
			"generatedAnswer", generatedAnswer
			);
		JSend jsend = new JSend(data);
		jsend.send(state);
	}

	private List findRelatedQuestion(String q) {
		// By Analogy to previous Q
		// Find related Qs
		List relatedQs = new RelatedQuestionFinder().run(q);
		
		findRelatedAnswer(relatedQs);
		return relatedQs;
	}

	private  List findRelatedAnswer(List relatedQs) {
		// TODO adapt related Q into an answer
		List relatedAs = new ArrayList();
		
		for (int i=0; i<relatedQs.size(); i++) {
			if (relatedQs!=null &&  ! relatedQs.isEmpty()) {
				Object rq = relatedQs.get(i);
				Object answer = SimpleJson.get(rq, "answers", 0);
				if (answer!=null ) {
					relatedAs.add(answer);
				}
			}
		}
		return relatedAs;
	}

	private Object generateAnswer(String q) throws UnsupportedEncodingException {
		Object answer = "";
		String pathToScript = System.getProperty("user.dir") 
				+ "/data-collection/lstmTestModelJepVersion.py";
		try {
			Jep jep = new Jep(new JepConfig().addSharedModules("tensorflow"));
			jep.runScript(pathToScript);
			answer = jep.getValue("generateResults('" + q + "')");
			jep.close();
		} catch (JepException e) {
			e.printStackTrace();
		}
		
		return answer;
	}
}
