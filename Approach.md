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


OCR is skipping small boxes from every page , mainly those which are one word per box 
merging ballon texts for now,received better results
updateing the texts span in merged box so to get better results
still missing on boxes,there are 83 lon unatched boxes ,
EasyOCR’s paragraph=True mode is dropping or scrambling text before alignment ever sees it. The diagnostic below checks that directly.
Taking boxes annotations for speakers and converting area arounf thsoe boxes to embeddings using CLIP
GOT BAD RESUTLS MADE A DIAGNOSTIC.PY TO EVALUATE- ITS RESULTS WERE-
The speaker model beats the floor, so the Phase 5 exam is technically met. But it recovers only about 20% of the available gap (0.085 of 0.416). Mean ARI of 0.13 says it is barely better than chance at grouping lines by speaker. The diagnose output explains why:

Panel features are dead. same_panel and panel_gap have importance 0.000, and the same-panel rate is exactly 0.0. Panel ids never reached the speaker data, so the strongest cue, turn-taking within a panel, is missing. This is the main fix.
page_gap dominates at 0.213. The model is mostly saying “balloons on the same page are probably the same person”. That is a weak prior, and with no panel information it has nothing better.
It under-splits. It predicts 2.09 speakers per sequence against 3.03 gold. The scorer weights every gold speaker equally, so missing minor speakers hurts.
CLIP similarity is nearly useless at 0.031 importance, as expected for black-and-white manga.

The fix is a self-contained panel finder, plus two turn-taking features, gap_parity and pp_gap. In dialogue, A, B, A, B means an even rank gap usually means the same speaker, and a tree model finds that easier if you give it explicitly.


Replacing pair_features with gpa parity and pp gap