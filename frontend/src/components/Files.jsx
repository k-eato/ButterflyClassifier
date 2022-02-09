import React, { useState } from 'react';
import { Heading } from "@chakra-ui/core";

export default function FileUploadPage(){
	const [selectedFile, setSelectedFile] = useState(null);
	const [isSelected, setIsSelected] = useState(false);
	const [prediction, setPrediction] = useState("");
	const [isSubmitted, setisSubmitted] = useState(false);

	// Load file data
	const changeHandler = (event) => {
		setSelectedFile(event.target.files[0]);
		setIsSelected(true);
		setisSubmitted(false);
		setPrediction("");
	};
	
	// Send image to TensorFlow model and receive predicted species
	const handleSubmission = async () => {
		const formData = new FormData();
		formData.append('file', selectedFile, selectedFile.name);
		const requestOptions = {
			method: 'POST',
			body: formData
		};
		await fetch('http://localhost:8000/upload', requestOptions);
		const response = await fetch("http://localhost:8000/result");
		const result = await response.json();
		setPrediction(result.data);
		setisSubmitted(true);
	}

	return(
   		<div style={{justifyContent: 'center', alignItems: 'center'}}>
			<Heading style={{textAlign: "center", backgroundColor: 'gray'}} as="h1">Butterfly Classifier</Heading>
			{isSelected ? (
				<div style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
					<img style={{padding: 10}} className="preview" src={URL.createObjectURL(selectedFile)} alt="" width="300" height="300"/>
				</div>
			) : (
				<Heading  size="s" style={{textAlign: "center"}}>Upload an image of a butterfly to classify it</Heading>
			)}
			<div style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
				<input style={{padding: 15}} type="file" name="file" onChange={changeHandler} />
			</div>
			{isSelected ? (
				<div>
					<div style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
						<button style={{padding: 15}} onClick={handleSubmission}>Submit</button>
					</div>
					{isSubmitted ? (
						<div>
							<Heading style={{textAlign: 'center'}}>Predicted Species:</Heading>
							<Heading style={{textAlign: 'center'}}  size="m">{prediction}</Heading>
						</div>
					) : (
						<div></div>
					)}
				</div>
			) : (
				<div></div>
			)}
		</div>
	)
}