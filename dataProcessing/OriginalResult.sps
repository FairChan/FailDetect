* Script created using the IEA IDB Analyzer (Version 5.0.50).
* Created on 12/17/2025 at 10:48 PM.
* Press Ctrl+A followed by Ctrl+R to submit this merge. 

include file = "C:\Users\ssema\AppData\Roaming\IEA\IDBAnalyzerV5\bin\Data\Templates\SPSS_Macros\IDBAnalyzer.ieasps".
include file = "C:\Users\ssema\Desktop\FailDetect\dataProcessing\IDBAnalyzerCountries.ieasps".

mcrComb 
	 indir="C:\Users\ssema\Downloads\dataset\eTPSI"/
	 infile=BCGAADZ7 BCGADUZ7 BCGAREZ7 BCGCHLZ7 BCGCOTZ7 BCGCQUZ7 BCGENGZ7 BCGFINZ7 BCGFRAZ7 BCGGEOZ7 BCGHKGZ7 BCGHUNZ7 BCGISRZ7 BCGITAZ7 BCGKORZ7 BCGLTUZ7 BCGMYSZ7 BCGNORZ7 BCGPRTZ7 BCGQATZ7 BCGRMOZ7 BCGRUSZ7 BCGSGPZ7 BCGSWEZ7 BCGTURZ7 BCGTWNZ7 BCGUSAZ7/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBCG/
	 keepVar=
       IDCNTRY 
       IDSCHOOL 
       IDPOP 
       IDGRADER 
       IDGRADE 
       ITLANG_C/
	 idbID='MZ7'.
EXECUTE.


mcrComb 
	 indir="C:\Users\ssema\Downloads\dataset\eTPSI"/
	 infile=BSGAADZ7 BSGADUZ7 BSGAREZ7 BSGCHLZ7 BSGCOTZ7 BSGCQUZ7 BSGENGZ7 BSGFINZ7 BSGFRAZ7 BSGGEOZ7 BSGHKGZ7 BSGHUNZ7 BSGISRZ7 BSGITAZ7 BSGKORZ7 BSGLTUZ7 BSGMYSZ7 BSGNORZ7 BSGPRTZ7 BSGQATZ7 BSGRMOZ7 BSGRUSZ7 BSGSGPZ7 BSGSWEZ7 BSGTURZ7 BSGTWNZ7 BSGUSAZ7/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBSG/
	 keepVar=
       IDCNTRY 
       IDBOOK 
       IDSCHOOL 
       IDCLASS 
       IDSTUD 
       IDPOP 
       IDGRADER 
       IDGRADE 
       ITSEX 
       ITADMINI 
       BSMMAT01 
       BSMMAT02 
       BSMMAT03 
       BSMMAT04 
       BSMMAT05 
       BSSSCI01 
       BSSSCI02 
       BSSSCI03 
       BSSSCI04 
       BSSSCI05/
	 idbID='MZ7'.
EXECUTE.


mcrComb 
	 indir="C:\Users\ssema\Downloads\dataset\eTPSI"/
	 infile=BSAAADZ7 BSAADUZ7 BSAAREZ7 BSACHLZ7 BSACOTZ7 BSACQUZ7 BSAENGZ7 BSAFINZ7 BSAFRAZ7 BSAGEOZ7 BSAHKGZ7 BSAHUNZ7 BSAISRZ7 BSAITAZ7 BSAKORZ7 BSALTUZ7 BSAMYSZ7 BSANORZ7 BSAPRTZ7 BSAQATZ7 BSARMOZ7 BSARUSZ7 BSASGPZ7 BSASWEZ7 BSATURZ7 BSATWNZ7 BSAUSAZ7/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBSA/
	 keepVar=
       IDCNTRY 
       IDBOOK 
       IDSCHOOL 
       IDCLASS 
       IDSTUD 
       IDPOP 
       IDGRADER 
       IDGRADE 
       ITSEX 
       ITADMINI 
       ILRELIAB 
       BSMMAT01 
       BSMMAT02 
       BSMMAT03 
       BSMMAT04 
       BSMMAT05 
       BSSSCI01 
       BSSSCI02 
       BSSSCI03 
       BSSSCI04 
       BSSSCI05/
	 idbID='MZ7'.
EXECUTE.


mcrComb 
	 indir="C:\Users\ssema\Downloads\dataset\eTPSI"/
	 infile=BSTAADZ7 BSTADUZ7 BSTAREZ7 BSTCHLZ7 BSTCOTZ7 BSTCQUZ7 BSTENGZ7 BSTFINZ7 BSTFRAZ7 BSTGEOZ7 BSTHKGZ7 BSTHUNZ7 BSTISRZ7 BSTITAZ7 BSTKORZ7 BSTLTUZ7 BSTMYSZ7 BSTNORZ7 BSTPRTZ7 BSTQATZ7 BSTRMOZ7 BSTRUSZ7 BSTSGPZ7 BSTSWEZ7 BSTTURZ7 BSTTWNZ7 BSTUSAZ7/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBST/
	 keepVar=
       IDCNTRY 
       IDBOOK 
       IDSCHOOL 
       IDCLASS 
       IDSTUD 
       IDTEALIN 
       IDTEACH 
       IDLINK 
       IDPOP 
       IDGRADER 
       IDGRADE 
       IDSUBJ 
       ITCOURSE 
       MATSUBJ 
       SCISUBJ 
       NMTEACH 
       NSTEACH 
       NTEACH 
       BSMMAT01 
       BSMMAT02 
       BSMMAT03 
       BSMMAT04 
       BSMMAT05 
       BSSSCI01 
       BSSSCI02 
       BSSSCI03 
       BSSSCI04 
       BSSSCI05 
       BSMIBM01 
       BSMIBM02 
       BSMIBM03 
       BSMIBM04 
       BSMIBM05 
       BSSIBM01 
       BSSIBM02 
       BSSIBM03 
       BSSIBM04 
       BSSIBM05 
       VERSION 
       SCOPE/
	 idbID='MZ7'.
EXECUTE.


mcrComb 
	 indir="C:\Users\ssema\Downloads\dataset\eTPSI"/
	 infile=BTMAADZ7 BTMADUZ7 BTMAREZ7 BTMCHLZ7 BTMCOTZ7 BTMCQUZ7 BTMENGZ7 BTMFINZ7 BTMFRAZ7 BTMGEOZ7 BTMHKGZ7 BTMHUNZ7 BTMISRZ7 BTMITAZ7 BTMKORZ7 BTMLTUZ7 BTMMYSZ7 BTMNORZ7 BTMPRTZ7 BTMQATZ7 BTMRMOZ7 BTMRUSZ7 BTMSGPZ7 BTMSWEZ7 BTMTURZ7 BTMTWNZ7 BTMUSAZ7/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBTM/
	 keepVar=
       IDCNTRY 
       IDSCHOOL 
       IDTEALIN 
       IDTEACH 
       IDLINK 
       ITCOURSE 
       IDPOP 
       IDGRADER 
       IDGRADE 
       IDSUBJ 
       ITLANG_T/
	 idbID='MZ7'.
EXECUTE.


mcrMOtO 
	 filedir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 file=tmpBSG tmpBSA/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBSGBSA/
	 mergeby=IDCNTRY idstud.
EXECUTE.


mcrMOtM 
	 filedir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 file=tmpBST/
	 tabledir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 table=tmpBTM/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBSTBTM/
	 mergeby=IDCNTRY idteach idlink.
EXECUTE.


mcrMOtM 
	 filedir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 file=tmpBSGBSA/
	 tabledir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 table=tmpBCG/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBCGBSGBSA/
	 mergeby=IDCNTRY idschool.
EXECUTE.


mcrMOtM 
	 filedir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 file=tmpBSTBTM/
	 tabledir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 table=tmpBCGBSGBSA/
	 outdir="C:\Users\ssema\Desktop\FailDetect\dataProcessing"/
	 outfile=tmpBCGBSGBSABSTBTM/
	 mergeby=IDCNTRY idstud.
EXECUTE.


select if (matwgt > 0).
ctylabls.
SAVE OUTFILE='C:\Users\ssema\Desktop\FailDetect\dataProcessing\OriginalResult.sav'.
EXECUTE.
host command = ['del "C:\Users\ssema\Desktop\FailDetect\dataProcessing\tmp*.sav"'].
NEW FILE.

