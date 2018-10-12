/**
 * A convenient place for ad-hoc widget tests.
 * This is not a replacement for proper unit testing - but it is a lot better than debugging via repeated top-level testing.
 */
import React from 'react';
import ReactDOM from 'react-dom';
import SJTest, {assert,assMatch} from 'sjtest';
import Login from 'you-again';
const MathJax = require('react-mathjax2');
import MDText from '../base/components/MDText';

const mathSplitter = (text) => {
	let r = /\$.+?\$/g;
	assMatch(text,String,"EgbotTestPage error: this is not a String");
	// non maths bits
	let bits1 = text.split(r);
	console.warn("bits1",bits1);
	let bits2 = [];
	// collect all the maths bits
	// NB: using replace but ignoring the result 
	text.replace(r, (a) => {
		// NB: we tried using regex groups but that throws the split higher up
		bits2.push(a.substring(1,a.length-1));		
	});
	// interleave them
	let bits = [];
	for(let i=0; i<bits1.length; i++) {
		bits.push(bits1[i]);
		bits.push(bits2[i]);
	}
	return bits;
};
window.mathSplitter = mathSplitter;

const MathParser = ({children}) => {
	let bits = mathSplitter(children);
	// TODO use this once weve sorted out the errors
	let res = bits.map( (e,i) => e ? (i%2===0 ? <MDText key={i}>{e}</MDText> : <MathJax.Context key={i}><MathJax.Node key={i}>{e}</MathJax.Node></MathJax.Context>) : null );

	return(
		<div>
			{res}
		</div>
	);
};

const TestPage = () => {
	let text = "Consider $x = y_1$ to be true";
	let text2 = "$\\frac{1}{27}$";

	return (
		<div className='TestPage'>
			<h2>Better Toast Page</h2>
			<p>Insert a test widget below</p>

			1. <MathParser>{text}</MathParser><br/>
			2. <MathParser>{text2}</MathParser><br/>
			3. <MathParser>$N$</MathParser><br/>
			4. <MathParser>$\sum x$</MathParser>
			
		</div>
	);
};

export default TestPage;
