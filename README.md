# Video-Watermarking-Using-Cuckoo-Search-Algorithm
The watermarking is implemented using <b>Cuckoo search algorithm</b> since it gives both robustness and imperceptibility to the extracted watermark. <b>Mantegna Levy Flight</b> and <b>McCulloch Levy Flight</b> are implemented along with <b>Cuckoo Search Algorithm</b>.<br>
The file <b>gui_mantegna.m</b> and <b>gui_mcculloch.m</b> are to be executed for Mantegna Levy Flight and McCulloch Levy Flight respectively. <br>

<h2>Steps happening in the entire watermarking process</h2>
<ol>
<li>At first the video is converted into frames.</li>
<li>Then scene change detection is done on each frame. When a scene change is detected, watermark is embbeded in the corresponding frame and the frame number is recored.</li>
<li>This step is continued until all the frames are scanned. At the end a key is generated in hexadecimal format by combining these frame numbers.</li>
<li>Then attacks are applied on the embedded frames and this process is authenticated by entering the key obtained in the previous step. 12 types of attacks are used for the extraction of watermark.</li>
<li>Finally all the frames are again converted back to video. Thus we get the watermaked video.</li>
</ol><br>

#Author
<i>Akash Choudhary</i>
