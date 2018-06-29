import React, { Component } from 'react';
import Misc from '../base/components/Misc';

class MainDiv extends Component {



	componentWillMount() {
	}

	render() {
		return (
			<div>
				<div className="container avoid-navbar">
					<div className="page MyPage">
						<Misc.Card>
							<div className="header">
								<div className="header-text">
									<p className="title"><span>EgBot</span></p>
									<p className="subtitle"><span>an example-based question & answer site</span><br/><span> for all of your math needs</span></p>
								</div>
							</div>
						</Misc.Card>		
						<Misc.Card>
							<div className="container">
								<div className="row">
									<div className="col-md-6 question-area-wrapper">
										<div className="input-group question-area">
											<div><b>Try out EgBot!</b></div><br/>
											<input placeholder="Type in your question" type="text" className="form-control"  aria-label="Type in your question" aria-describedby="question-input"/>
											<div className="input-group-append">
												<span className="input-group-text" id="question-input"><button className="btn btn-outline-secondary" type="button">Search</button></span>
											</div>
										</div>
										<br/>
										<textarea>Result will appear here</textarea>
									</div>
									<div className="col-md-6 play">
										<img className="play-demo" src="https://i.imgur.com/ZR17qnE.png"/>
									</div>
								</div>
							</div>
						</Misc.Card>	
					</div>
				</div>
			</div>
		);
	} // ./render()
} // ./MainDiv

export default MainDiv;
