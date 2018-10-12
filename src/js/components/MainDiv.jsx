import React, { Component } from 'react';
import { assert } from 'sjtest';
import Login from 'you-again';
import md5 from 'md5';
import { modifyHash } from 'wwutils';
// Plumbing
import JSend from '../base/data/JSend';
import DataStore from '../base/plumbing/DataStore';
import Roles from '../base/Roles';
import C from '../C';
// import ServerIO from '../plumbing/ServerIO';
// import ActionMan from '../plumbing/ActionMan';
// Widgets
import MessageBar from '../base/components/MessageBar';
import NavBar from '../base/components/NavBar';
import LoginWidget from '../base/components/LoginWidget';
import Misc from '../base/components/Misc';

import CardAccordion, {Card} from '../base/components/CardAccordion';
import PropControl from '../base/components/PropControl';
import BS3 from '../base/components/BS3';

// Pages
import BasicAccountPage from '../base/components/AccountPageWidgets';
import E404Page from '../base/components/E404Page';
import TestPage from './EgbotTestPage';

import MathWidget from './MathWidget';

class MainDiv extends Component {

	componentWillMount() {
		// redraw on change
		const updateReact = (mystate) => this.setState({});
		DataStore.addListener(updateReact);

		Login.app = C.app.service;
		// Set up login watcher here, at the highest level		
		Login.change(() => {
			// ?? should we store and check for "Login was attempted" to guard this??
			if (Login.isLoggedIn()) {
				// close the login dialog on success
				LoginWidget.hide();
			} else {
				// poke React via DataStore (e.g. for Login.error)
				DataStore.update({});
			}
			this.setState({});
		});

		// Are we logged in?
		Login.verify();
	}

	componentDidCatch(error, info) {
		// Display fallback UI
		this.setState({error, info, errorPath: DataStore.getValue('location', 'path')});
		console.error(error, info); 
		if (window.onerror) window.onerror("Caught error", null, null, null, error);
	}

	render() {
		if ((""+window.location).indexOf("test") !== -1) {
			return <TestPage />;
		}	
		return (
			<div className="container avoid-navbar">
				<div className="page MyPage">
					<Card>
						<div className="header">
							<div className="header-text">
								<p className="title"><span>EgBot</span></p>
								<p className="subtitle"><span>an example-based question & answer site</span><br/><span> for all of your math needs</span></p>
							</div>
						</div>
					</Card>		
					<Card>
						<div className="row">
							<div className="col-md-4 question-area-wrapper">
								<div className="input-group question-area">
									<div><b>Type in your question</b></div>
									<QuestionForm />
								</div>
							</div>
							<div className="col-md-4 play">
								<div><b>See some similar Q&A's</b></div>
								<SimilarAnswerPanel />
							</div>
							<div className="col-md-4 play">
								<div><b>See what EgBot says</b></div>
								<EgBotAnswerPanel />
							</div>
						</div>
					</Card>	
				</div>
			</div>
		);
	} // ./render()
} // ./MainDiv


const qpath = ['widget','qform'];

const apath = () => {
	let qform = DataStore.getValue(qpath) || {};
	let qhash = qform.q? md5(qform.q) : 'blank';
	let _apath = ['widget', 'askResponse', qhash];	
	return _apath;
};

const QuestionForm = () => {

	const onEnterPress = (e) => {
		if(e.keyCode === 13 && e.shiftKey === false) {
			e.preventDefault();
			document.querySelector(".question-button").click();
		}
	};

	return (
		<div>
			<PropControl 
				path={qpath} prop='q'
				type="textarea" 
				label=""
				rows="3"
				onKeyDown={onEnterPress}
			/>
			<div className="btn-group" role="group" aria-label="">
				<Misc.SubmitButton path={qpath} className="btn btn-primary question-button" url='/ask' responsePath={apath()}>Ask</Misc.SubmitButton>
			</div>
		</div>
	);
};

let _handleClick = (carouselPosition, carouselTotal) => {
	// find the next q&a pair to show, starting from the beginning when reaching the end
	carouselPosition = (carouselPosition+1)%carouselTotal;
	DataStore.setValue(['widget', 'similarAnswerPanel', 'carouselPosition'], carouselPosition);
};

const SimilarAnswerPanel = () => {

	let askResponse = DataStore.getValue(apath()) || {};
	console.log(askResponse);
	if ( ! askResponse.relatedAs) {
		return (<div className='well'></div>
		);
	}

	//MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MathExample"]);
	let carouselTotal = askResponse.relatedQs.length;
	let carouselPosition = DataStore.getValue(['widget', 'similarAnswerPanel', 'carouselPosition']) || 0; // goes through the q&a pairs to display one at a time
	let relatedQs = askResponse.relatedQs[carouselPosition].body_markdown; 
	let relatedAs = askResponse.relatedAs[carouselPosition]; 

	return (<div className='well'>
		<div className='qa-question'>
			<div><b>Question</b></div> 
			<div><MathWidget>{relatedQs}</MathWidget></div>
		</div><br/>
		<div className='qa-answer'>
			<div><b>Answer</b></div>
			<div><MathWidget>{relatedAs.body_markdown || relatedAs}</MathWidget>></div>
		</div><br/>
		<div>
			<button type="button" className="btn btn-default question-button" onClick={() => _handleClick(carouselPosition, carouselTotal)}>Next</button>
		</div>
	</div>);
};

const EgBotAnswerPanel = () => {
	let askResponse = DataStore.getValue(apath()) || {};

	if ( ! askResponse.generatedAnswer) {
		return (<div className='well'></div>);
	}
	let generatedAnswer = askResponse.generatedAnswer; 
	return (<div className='well'>
		<div className='qa-answer'>
			<div>{generatedAnswer || 'Sorry, I don\'t know'}</div>
		</div>
	</div>);
};

export default MainDiv;
