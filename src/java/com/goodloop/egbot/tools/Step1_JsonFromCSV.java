package com.goodloop.egbot.tools;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.eclipse.jetty.util.ajax.JSON;
//import org.python.core.PyObject;
//import org.python.core.PyString;
//import org.python.util.PythonInterpreter;

import com.winterwell.gson.GsonBuilder;
import com.winterwell.utils.IReplace;
import com.winterwell.utils.MathUtils;
import com.winterwell.utils.StrUtils;
import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.io.CSVReader;
import com.winterwell.utils.io.CSVWriter;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
/**
 * Convert .csv to json - 
 * ...Handling Python objects within the csv.
 * 
 * Assumes:
 *  - You have .csv files of MathStackExchange (MSE) questions, as created by the python scripts.
 *  - These are in data/raw
 * 
 * The main method will convert all .csv files in data/raw, putting outputs into data/build
 * 
 * @author daniel
 *
 */
public class Step1_JsonFromCSV {

	public static void main(String[] args) throws IOException {
		// all files
		File rawDataDir = new File("data/raw");
		File buildDataDir = new File("data/build");
		buildDataDir.mkdirs();
		for(File f : rawDataDir.listFiles()) {
			if ( ! f.getName().endsWith(".csv")) continue;
			File fout = new File(buildDataDir, FileUtils.changeType(f, "json").getName());
			convertToJson(f, fout);
		}
	}

	private static void convertToJson(File f, File fout) throws IOException {
		Log.i("jsonfromcsv", "Converting "+f+" ...");
		CSVReader r = new CSVReader(f);
		BufferedWriter w = FileUtils.getWriter(fout);
		w.write("[\n	");
		String[] cols = r.next();
		boolean first = true;
		int cnt = 0;
		for (String[] row : r) {
			if ( ! first) w.write(",\n	");
			first = false;
			// make json obj
			ArrayMap jobj = new ArrayMap();
			for(int i=0; i<cols.length; i++) {
				if (i==row.length) break;
				String k = cols[i];
				if (Utils.isBlank(k)) k = "col_"+i;
				String v = row[i];
				// json object?
				Object vobj = convertCell(v);
				jobj.put(k, vobj);
			}
			// write it out
			String json = JSON.toString(jobj);
			w.write(json);
			cnt++;
		}
		w.write("\n]\n");
		r.close();
		w.close();
		Log.i("jsonfromcsv", "...wrote "+cnt+" to "+fout);
	}

//	// throws an exception "ImportError: Cannot import site module"
//	static PythonInterpreter interpreter;
//	static {
//
//		Properties preprops = System.getProperties();		
//		Properties props = new Properties();
//		props.setProperty("python.import.site", "false");
//		PythonInterpreter.initialize(preprops, props, new String[0]);
//
//		interpreter = new PythonInterpreter();
//	}
	
	/**
	 * Detect single \s (not allowed in json ??except as unicode markers)
	 * 
	 * TODO json explicit unicode support??
	 * TODO also detect at start/end string 
	 */
	static Pattern UNESCAPED_SLASH = Pattern.compile("[^\\\\]\\\\[^\\\\]");
	
	private static Object convertCell(String v0) {
		String v = v0;
		try {			
			if (Utils.isBlank(v)) return null;
			
			// _Someone_ (well: Irina) stored Python and ad-hoc formatted strings instead of json
			// convert some Python strings into standard json
			if (v.contains("u'") || v.contains("u\"")
					|| UNESCAPED_SLASH.matcher(v).find()) 
			{
				v = convertPythonToJson(v);
			}
			Pattern array = Pattern.compile("^\\[\\w");
			if (array.matcher(v).find()) {
				String[] tags = v.substring(1, v.length()-1).split(",");
//				v = JSON.toString(tags);
				return tags;
			}
			
			if (v.startsWith("[") || v.startsWith("{")) {
				Object vobj = JSON.parse(v);
				return vobj;
			}
			if (MathUtils.isNumber(v)) {
				double nv = MathUtils.toNum(v);
				if (nv == Math.round(nv)) return (int) nv;
				return nv;
			}
			return v;
		} catch(Exception ex) {
			Log.e(v+"\n\n"+v0, ex);
			return "";
		}
	}

	private static String convertPythonToJson(String v) {
//		interpreter.eval("\ndef to_json(d):\n	d;\n"); // json.dumps(ast.literal_eval(d))
//		PyObject ToJson = interpreter.get("to_json");
//		PyObject result = ToJson.__call__(new PyString(v));
//		String vjson = (String) result.__tojava__(String.class);
		
		// HACK
		String vjson = v.replaceAll("u'", "\"");
		vjson = vjson.replaceAll("'([:,{}\\]\\[])", "\"$1");
		vjson = vjson.replaceAll("False", "false");
		vjson = vjson.replaceAll("True", "true");
		
		vjson = vjson.replaceAll("\\\\u", "\\\\\\\\u");
		vjson = vjson.replaceAll("\\\\x", "\\\\\\\\x");
//		vjson = StrUtils.replace(vjson, Pattern.compile("&#"), (sb, match) -> {
//			System.out.println(match.group());
//		});
//		for(int i=0; i<v.length(); i++) {
//			char c = v.charAt(i);
//		}
		return vjson;
	}
}
