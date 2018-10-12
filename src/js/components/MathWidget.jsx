import React from 'react';
import ReactDOM from 'react-dom';
import SJTest, {assert,assMatch} from 'sjtest';
const MathJax = require('react-mathjax2');
import MDText from '../base/components/MDText';

const mathSplitter = (text) => {
	let r = /\$+.+?\$+/g;
	assMatch(text,String,"MathWidget error: this is not a String");
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

const MathWidget = ({children}) => {
	console.log(MathJax);
	let bits = mathSplitter(children);
	// TODO use this once weve sorted out the errors
	let res = bits.map( (e,i) => e ? (i%2===0 ? <MDText source={e} key={i} /> : <MathJax.Context key={i}><MathJax.Node inline key={i}>{e}</MathJax.Node></MathJax.Context>) : null );
                                            
	return(
		<div className='MathWidget'>
			{res}
		</div>
	);
};

export default MathWidget;
