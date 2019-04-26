package com.goodloop.egbot.data;

import java.util.HashMap;

public class PauliusQuestion  extends HashMap<String, Object> {

	public String getId() {
		Object qid = get("question_id");
		String s = qid.toString();
		if (qid instanceof Number && s.endsWith(".0")) {
			s = s.substring(0, s.length()-2);
		}
		return s;
	}
}
