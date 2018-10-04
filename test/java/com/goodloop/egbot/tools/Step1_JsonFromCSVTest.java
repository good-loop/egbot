package com.goodloop.egbot.tools;

import static org.junit.Assert.*;

import java.util.regex.Matcher;

import org.junit.Test;

public class Step1_JsonFromCSVTest {

	@Test
	public void testUnescapedSlash() {
		{	// escaped
			String s1 = "foo bar \n \\\\' ";
			Matcher m = Step1_JsonFromCSV.UNESCAPED_SLASH.matcher(s1);
			boolean f = m.find();
			assert ! f;
		}
		{	// unescaped
			String s1 = "foo bar $\\d + \\blah$";
			Matcher m = Step1_JsonFromCSV.UNESCAPED_SLASH.matcher(s1);
			boolean f = m.find();
			assert f;
		}
	}
	

}
