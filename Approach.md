Started by discussing it with ChatGPT to understand clearly about whats been asked and what to do
Saw the given dataset structure and samples
Tried to read score.py,someweher successful,gave it to chat gpt to brief it out and explain technicality 
asked it to make a sample roadmap and an architecutre file which would have techstack and workflow info
went to claude to review everthing and ask for updates

*What they suggested-*
they gave a good head start about datachecking,training methods (k-folds and experimentations), alignment system,ocr training and recommendation,filtering system etc

Ran into problem with alignment system-
tried using self made algo,open cv manga comic and faster rcnn 
now training yolo for detecting bubbles and boxes
IDEA IS-
train yolo for bubbles and boxes 
then use ocr to get text from thosee and then a filtering system that helps get the spoken text only and then character assignment

Why not YOLO for alignment? YOLO needs box labels to train, and getting box labels is exactly the problem. Alignment doesn't need a trained detector. A pretrained detector and OCR find the boxes, and text matching links each one to a label line. After alignment you'll have pseudo-box labels, so a detector could be fine-tuned later if it turns out to be the weak point. That is optional.

Why not OpenCV? Alignment is a text-matching problem, so image processing doesn't help with it. I'm not aware of a specific "manga edition" of OpenCV. Plain OpenCV is useful for panel detection (contours). For text boxes, use a pretrained detector such as PaddleOCR's or comic-text-detector (check this one is still available).

Turned score.py into a module to use its functions
Using score.py as a module and then importing it funtions and testing methods to improve our work

need to fix ocr stuff to scoremax / using regex based algo for that


OCR model choosing--easy ocr,paddle ocr and florence,not using paddle as it works on cpu not gpu
removed florence too,had some isseus while running,using just easy ocr, and easyocr performs well 