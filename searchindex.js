Search.setIndex({"docnames": ["exo1_cancer", "exo1_housing", "exo2_fraud", "exo2_wind", "exo3_fraud_preprocessing", "exo3_fraud_training_bs", "exo3_fraud_training_gs", "intro"], "filenames": ["exo1_cancer.ipynb", "exo1_housing.ipynb", "exo2_fraud.ipynb", "exo2_wind.ipynb", "exo3_fraud_preprocessing.ipynb", "exo3_fraud_training_bs.ipynb", "exo3_fraud_training_gs.ipynb", "intro.md"], "titles": ["Exercise 1: Breast cancer", "Exercise 1: Boston Housing", "Exercise 2: Fraud detection", "Exercise 2: Wind speed", "Exercise 3: Fraud detection preprocessing", "Exercise 3: Fraud detection training (Bayes Optimization)", "Exercise 3: Fraud detection training", "Partial 3"], "terms": {"thi": [0, 2, 3], "i": [0, 1, 2, 3, 4, 5, 6], "small": [], "sampl": [2, 4, 5, 6], "give": [], "you": [], "feel": [], "how": [], "content": [], "structur": [], "It": [], "show": [0, 1, 2, 3, 5, 6], "off": [], "few": 3, "major": 3, "file": [], "type": [2, 3], "well": [], "some": [2, 3, 4], "doe": [], "go": [3, 4], "depth": [], "ani": [], "particular": 3, "topic": [], "check": [], "out": [], "document": [], "more": [0, 2, 4], "inform": [], "page": [], "bundl": [], "see": [2, 3, 4], "import": [0, 1, 2, 3, 4, 5, 6], "panda": [0, 1, 2, 3, 4, 5, 6], "pd": [0, 1, 2, 3, 4, 5, 6], "df_wind": 3, "read_csv": [2, 3, 4], "data_wind": 3, "csv": [2, 3, 4], "head": [2, 3, 4], "hora": 3, "utc": 3, "vento": 3, "dire\u00ef": 3, "\u00bd\u00ef": 3, "\u00bdo": 3, "horaria": 3, "gr": 3, "\u00ef": 3, "\u00bd": 3, "velocidad": 3, "m": [2, 3, 4], "": [0, 1, 2, 3, 4, 5, 6], "umidad": 3, "rel": 3, "max": [1, 2, 3], "na": 3, "ant": 3, "aut": 3, "min": [2, 3], "temperatura": 3, "m\u00ef": 3, "\u00bdxima": 3, "\u00bdc": 3, "\u00bdnima": 3, "relativa": 3, "do": 3, "ar": [2, 3, 4], "pressao": 3, "atmosferica": 3, "ao": 3, "nivel": 3, "da": 3, "estacao": 3, "mb": 3, "precipita\u00ef": 3, "total": [0, 1, 3], "hor\u00ef": 3, "\u00bdrio": 3, "mm": 3, "rajada": 3, "maxima": 3, "press\u00ef": 3, "0": [0, 1, 2, 3, 4, 5, 6], "12": [2, 3, 5, 6], "00": 3, "809017": 3, "1": [2, 3, 4, 5, 6, 7], "8": [0, 2, 3, 5, 6], "69": [2, 3], "60": [2, 3], "22": [2, 3], "6": [0, 2, 3, 4, 5, 6], "20": [0, 1, 2, 3, 4, 5, 6], "7": [1, 2, 3, 6], "61": [2, 3], "888": 3, "3": [0, 1, 2, 3], "887": 3, "13": [2, 3], "965926": 3, "62": [2, 3], "55": [2, 3], "24": [2, 3], "5": [0, 1, 2, 3, 4, 5, 6], "4": [0, 1, 2, 3, 4, 5, 6], "14": [2, 3], "891007": 3, "56": 3, "50": [2, 3, 4, 5, 6], "25": [2, 3, 5, 6], "51": 3, "9": [0, 2, 3], "15": [2, 3, 5, 6], "848048": 3, "52": [2, 3], "44": [2, 3], "27": [2, 3], "16": [0, 2, 3, 5, 6], "224951": 3, "43": [2, 3], "46": 3, "886": 3, "info": [2, 3], "class": [2, 3], "core": [2, 3], "frame": [2, 3], "datafram": [0, 1, 2, 3, 4, 5, 6], "rangeindex": [2, 3], "87693": 3, "entri": [2, 3], "87692": 3, "column": [0, 1, 2, 3, 4, 5, 6], "non": [2, 3], "null": 3, "count": [2, 3], "dtype": [0, 2, 3, 4], "object": [0, 2, 3, 4], "float64": [2, 3, 4], "10": [0, 1, 2, 3, 4, 5, 6], "11": [2, 3], "memori": [2, 3], "usag": [2, 3], "isnul": [2, 3], "sum": [2, 3], "we": [0, 2, 3, 4], "don": [2, 3, 4], "t": [2, 3, 4], "have": [2, 3, 4], "miss": [2, 3], "valu": [1, 2, 3, 4], "index": [0, 1, 3, 4, 5, 6], "signif": 3, "name": 3, "so": [0, 2, 3], "can": [2, 3, 4], "renam": 3, "them": [2, 3, 4], "new_column": 3, "hour": 3, "direct": 3, "humid": 3, "temperatur": 3, "atmospher": 3, "pressur": 3, "station": 3, "level": 3, "precipit": 3, "gust": 3, "dict": 3, "zip": 3, "inplac": [2, 3, 4], "true": [0, 1, 2, 3, 4, 5, 6], "instanc": 3, "sometim": 3, "17": [2, 3], "19": [3, 6], "without": [0, 3], "18": [2, 3], "add": 3, "later": 3, "try": [3, 4], "imput": [2, 3], "theses": 3, "total_ad": 3, "_": [0, 3, 4, 5, 6], "rang": [0, 1, 2, 3, 6], "23": [2, 3], "idx": 3, "nb_ad": 3, "while": [2, 3], "len": [2, 3, 4, 5, 6], "cur_hour": 3, "int": [2, 3], "split": 3, "next_hour": 3, "new_str": 3, "str": 3, "els": [3, 4, 5, 6], "new_row": 3, "concat": [3, 4], "iloc": 3, "ignore_index": 3, "print": [0, 1, 2, 3, 4, 5, 6], "f": [0, 1, 2, 3, 4, 5, 6], "ad": 3, "row": [2, 3, 4], "866": 3, "332": [2, 3], "241": 3, "182": [2, 3], "141": [2, 3], "122": [2, 3, 5], "106": 3, "92": 3, "79": 3, "67": [2, 3], "57": [2, 3], "41": [2, 3], "37": [2, 3], "31": [2, 3], "2583": 3, "The": [0, 1, 2, 3], "given": 3, "string": 3, "consid": 3, "almost": [0, 2, 3], "ten": 3, "year": 3, "chang": 3, "format": 3, "start": 3, "2000": [3, 4], "datetim": 3, "dt": 3, "start_dat": 3, "date": 3, "timedelta": 3, "unit": 3, "intern": 3, "system": 3, "appli": [3, 4], "lambda": [2, 3], "x": [1, 2, 3, 4, 5, 6], "100": [0, 1, 2, 3, 5, 6], "drop": [0, 2, 3, 4, 5, 6], "k": [0, 1, 3, 6], "273": 3, "pa": 3, "1000": [0, 1, 2, 3, 5, 6], "sort_valu": [2, 3, 4, 5, 6], "01": [0, 1, 3, 6], "293": 3, "85": [0, 3], "295": 3, "75": [2, 3], "88820": 3, "88770": 3, "65": 3, "297": 3, "35": 3, "88840": 3, "45": [2, 3], "298": 3, "88810": 3, "300": [2, 3], "88740": 3, "88650": 3, "to_csv": 3, "data_wind_eda": 3, "fals": [0, 1, 2, 3, 4, 5, 6], "data_eda": [2, 3], "shape": [0, 1, 2, 3, 4, 5, 6], "90276": 3, "missing_data_count": [2, 3], "missing_data_percentag": [2, 3], "round": [0, 1, 2, 3, 4, 5, 6], "missing_data_stat": [2, 3], "By": 3, "now": [2, 3], "around": [2, 3], "describ": [2, 3], "000000": [2, 3], "mean": [2, 3, 4], "405810": 3, "466192": 3, "161076": 3, "690585": 3, "631762": 3, "834570": 3, "071264": 3, "661467": 3, "88725": 3, "192547": 3, "000161": 3, "88758": 3, "072366": 3, "88689": 3, "109279": 3, "std": [2, 3], "686247": 3, "313968": 3, "311157": 3, "196402": 3, "201663": 3, "513744": 3, "721386": 3, "199923": 3, "401": 3, "240375": 3, "001308": 3, "364": 3, "675005": 3, "356": 3, "453928": 3, "120000": 3, "100000": 3, "281": 3, "550000": 3, "282": 3, "350000": 3, "86340": 3, "86530": 3, "86280": 3, "156434": 3, "500000": [2, 3], "400000": 3, "540000": 3, "480000": 3, "291": 3, "292": 3, "510000": 3, "88530": 3, "88560": 3, "88500": 3, "788011": 3, "720000": 3, "640000": 3, "294": 3, "680000": 3, "88720": 3, "88750": 3, "88690": 3, "970296": 3, "800000": 3, "870000": 3, "296": 3, "250000": 3, "850000": 3, "840000": 3, "88910": 3, "88930": 3, "88880": 3, "300000": 3, "980000": 3, "307": 3, "308": 3, "450000": 3, "990000": 3, "102350": 3, "070800": 3, "91310": 3, "91090": 3, "plotli": [3, 4], "graph_object": [3, 4], "col": [2, 3, 4], "fig": [0, 1, 2, 3, 4, 5, 6], "figur": [3, 4, 5, 6], "add_trac": [3, 4], "scatter": [1, 3, 4], "y": [1, 2, 3, 4, 5, 6], "mode": [3, 4], "line": [3, 4], "update_layout": [3, 4], "titl": [2, 3, 4, 5, 6], "over": 3, "time": [0, 1, 3, 4, 5, 6], "xaxis_titl": [3, 4], "yaxis_titl": [3, 4], "margin": 3, "l": 3, "r": [3, 4, 5, 6], "40": [2, 3], "b": 3, "ssl": [], "_create_default_https_context": [], "_create_unverified_context": [], "lot": [2, 3, 4], "veri": [2, 3], "fast": [0, 3], "which": [0, 2, 3, 4], "make": 3, "hard": 3, "predict": [0, 1, 3, 4, 5, 6], "field": 3, "cycl": 3, "dure": 3, "dai": [0, 3], "assum": 3, "special": 3, "climat": 3, "outlier": [2, 3], "atmosfer": 3, "becaus": [0, 1, 3], "valid": 3, "keep": [3, 4], "doesn": [2, 3], "influenc": 3, "all": [2, 3, 4], "onli": 3, "ones": 3, "notic": [2, 3], "high": [2, 3, 4], "variabl": [2, 3, 4], "imagin": 3, "sensor": 3, "period": 3, "matplotlib": [0, 1, 2, 3, 5, 6], "pyplot": [0, 1, 2, 3, 5, 6], "plt": [0, 1, 2, 3, 5, 6], "seaborn": [0, 2, 3, 5, 6], "sn": [0, 2, 3, 5, 6], "ax": [0, 1, 2, 3, 5, 6], "subplot": [0, 1, 2, 3, 5, 6], "figsiz": [0, 1, 2, 3, 5, 6], "30": [2, 3], "flatten": [2, 3], "var": [2, 3], "enumer": [0, 1, 2, 3, 4, 5, 6], "histplot": [2, 3], "kde": [2, 3], "set_titl": [0, 1, 2, 3, 5, 6], "histogram": [2, 3], "tight_layout": [2, 3], "reveal": [2, 3], "uniform": 3, "distribut": [2, 3], "cannot": [2, 3], "perform": [2, 3], "normal": 3, "test": [3, 5, 6], "probabl": 3, "repartit": [2, 3], "howev": 3, "look": [2, 3], "mayb": 3, "boxplot": [2, 3], "set_xlabel": [1, 2, 3], "For": [2, 3], "concentr": 3, "rain": 3, "pairplot": [2, 3], "diag_kind": [2, 3], "axisgrid": [2, 3], "pairgrid": [2, 3], "0x79b2090965e0": [], "error": [], "callback": [], "function": [], "_draw_all_if_interact": [], "0x79b246e0c700": [], "post_execut": [], "argument": [], "arg": [], "kwarg": [], "keyboardinterrupt": [], "traceback": [], "most": [0, 2, 4], "recent": [], "call": [], "last": 2, "miniconda3": [], "env": [], "ml_venv": [], "lib": [], "python3": [], "site": [], "packag": [], "py": [], "197": 2, "195": [], "def": [], "none": [0, 1, 5, 6], "196": [], "is_interact": [], "draw_al": [], "_pylab_help": [], "132": [], "gcf": [], "cl": [], "forc": [], "130": 2, "manag": [], "get_all_fig_manag": [], "131": [], "canva": [], "stale": [], "draw_idl": [], "backend_bas": [], "1893": [], "figurecanvasbas": [], "self": [], "1891": [], "_is_idle_draw": [], "1892": [], "_idle_draw_cntx": [], "draw": [], "backend": [], "backend_agg": [], "388": [], "figurecanvasagg": [], "385": [], "acquir": [], "lock": [], "share": [], "font": [], "cach": [], "386": [], "toolbar": [], "_wait_cursor_for_draw_cm": [], "387": [], "nullcontext": [], "render": [], "389": [], "A": 2, "gui": [], "mai": [], "need": 2, "updat": [], "window": [], "us": [0, 2], "390": [], "forget": [], "superclass": [], "391": [], "super": [], "artist": [], "95": 2, "_finalize_raster": [], "local": [], "draw_wrapp": [], "93": 2, "wrap": [], "94": 2, "result": [5, 6], "96": [5, 6], "_raster": [], "97": 2, "stop_raster": [], "72": 2, "allow_raster": [], "get_agg_filt": [], "70": 2, "start_filt": [], "return": [], "73": [], "final": [], "74": 2, "3154": [], "3151": [], "valueerror": [], "occur": [], "when": [], "resiz": [], "3153": [], "patch": [], "mimag": [], "_draw_list_compositing_imag": [], "3155": [], "suppresscomposit": [], "3157": [], "sfig": [], "subfig": [], "3158": [], "imag": [], "parent": [], "suppress_composit": [], "not_composit": [], "has_imag": [], "133": 2, "134": [], "composit": [], "adjac": [], "togeth": [], "135": 6, "image_group": [], "_base": [], "3070": [], "_axesbas": [], "3067": [], "artists_raster": [], "3068": [], "_draw_raster": [], "3071": [], "3073": [], "close_group": [], "3074": [], "collect": [5, 6], "1005": [], "_collectionwiths": [], "1002": [], "1003": [], "1004": [], "set_siz": [], "_size": [], "dpi": [], "408": [], "406": [], "gc": [], "set_antialias": [], "_antialias": [], "407": [], "set_url": [], "_url": [], "draw_mark": [], "409": [], "path": [4, 5, 6], "combined_transform": [], "frozen": [], "410": [], "mpath": [], "offset": [], "offset_trf": [], "tupl": [], "facecolor": [], "411": [], "412": [], "_gapcolor": [], "413": [], "first": 4, "within": [], "gap": [], "observ": [2, 3], "linear": [0, 1, 2, 3, 4, 5, 6], "correl": [2, 3, 4], "between": [2, 3, 4], "featur": [3, 4, 5, 6], "like": [2, 3], "even": 3, "exact": 3, "also": [2, 3], "underlin": 3, "extrem": 3, "exempl": [2, 3], "our": [2, 3], "output": [3, 4], "corr_matrix": 3, "corr": [2, 3], "ab": [2, 3, 4], "heatmap": [0, 2, 3, 5, 6], "annot": [0, 2, 3, 5, 6], "cmap": [2, 3], "coolwarm": [2, 3], "fmt": [0, 2, 3, 5, 6], "2f": [2, 3], "matrix": [2, 3], "confirm": [2, 3], "same": [2, 3], "help": 3, "train": [2, 3, 4, 7], "reduc": [2, 3], "multicolinear": 3, "exercis": 7, "2": [0, 1, 4, 5, 7], "wind": 7, "speed": 7, "put": [], "second": [], "sort": [2, 3], "rest": [], "0x74985fe3f0a0": 3, "from": [0, 1, 2, 3, 4, 5, 6], "statsmodel": 3, "graphic": 3, "tsaplot": 3, "plot_acf": 3, "dropna": [2, 3], "lag": 3, "set_ylim": 3, "autocorrel": 3, "month": 3, "degre": 3, "auto": 3, "week": 3, "In": [0, 3], "peak": 3, "everi": 3, "except": 3, "other": [2, 3], "moreov": [2, 3], "tsa": 3, "stattool": 3, "adful": 3, "test_adful": 3, "p": 3, "stationar": 3, "affirm": 3, "confid": 3, "ha": [0, 1, 3], "tendenc": 3, "detect": [0, 7], "want": 0, "minim": 0, "incorrect": 0, "neg": 0, "e": 0, "sick": 0, "patient": 0, "classifi": 0, "diagnosi": 0, "could": [0, 2], "lead": 0, "seriou": 0, "health": 0, "problem": 0, "therefor": 0, "metric": [0, 5, 6], "applic": 0, "recal": [0, 4, 5, 6], "limit": 0, "number": [0, 2, 4], "sklearn": [0, 1, 4, 5, 6], "dataset": [0, 1, 2], "load_breast_canc": 0, "target_nam": 0, "arrai": 0, "malign": 0, "benign": 0, "u9": 0, "target": [0, 2], "safe": 0, "model_select": [0, 1, 4, 6], "train_test_split": [0, 1, 4], "df_cancer": 0, "feature_nam": 0, "x_train": [0, 1, 4, 5, 6], "x_test": [0, 1, 4, 5, 6], "y_train": [0, 1, 4, 5, 6], "y_test": [0, 1, 4, 5, 6], "axi": [0, 2, 4, 5, 6], "test_siz": [0, 1, 4], "random_st": [0, 1, 2, 4, 5, 6], "42": [0, 1, 2, 4, 5, 6], "standardscal": [0, 1, 5, 6], "scaler": [0, 1, 5, 6], "x_train_sc": 0, "fit_transform": 0, "x_test_sc": 0, "transform": [0, 4], "svm": [0, 1, 5, 6], "svc": [0, 5, 6], "minmaxscal": [0, 1, 5, 6], "nb_col": [0, 1, 5, 6], "dict_model": [0, 1, 5, 6], "name_clf": [0, 1, 5, 6], "poli": [0, 1, 5, 6], "model": [0, 1, 2, 5, 6], "kernel": [0, 1, 2, 5, 6], "grid": [0, 1, 5, 6], "model__c": [0, 1, 5, 6], "model__gamma": [0, 1], "rbf": [0, 1, 5, 6], "sigmoid": [0, 1, 5, 6], "take": [0, 2], "than": [0, 2, 4], "run": 0, "choos": 0, "por": 0, "pipelin": [0, 1, 4, 5, 6], "gridsearchcv": [0, 1, 6], "util": [0, 1, 2, 4, 5, 6], "model_evaluation_clf": [0, 4, 5, 6], "warn": [0, 1, 4, 5, 6], "filterwarn": [0, 1, 4, 5, 6], "ignor": [0, 1, 4, 5, 6], "cpu": [0, 1, 4, 5, 6], "accuraci": [0, 4, 5, 6], "precis": [0, 4, 5, 6], "f1": [0, 4, 5, 6], "score": [0, 1, 4, 5, 6], "auc": [0, 4, 5, 6], "nb_re": [0, 1, 4, 5, 6], "dict_clf": [0, 1, 5, 6], "model_nam": [0, 1, 5, 6], "step": [0, 1, 4, 5, 6], "param_grid": [0, 1, 5, 6], "clf": [0, 1, 5, 6], "cv": [0, 1, 5, 6], "n_job": [0, 1, 5, 6], "verbos": [0, 1, 5, 6], "start_tim": [0, 1, 4, 5, 6], "fit": [0, 1, 4, 5, 6], "end_tim": [0, 1, 4, 5, 6], "best": [0, 1, 5, 6], "param": [0, 1, 5, 6], "n": [0, 1, 5, 6], "best_params_": [0, 1, 5, 6], "eval": [0, 1, 4, 5, 6], "loc": [0, 1, 4, 5, 6], "roc_auc": [0, 4, 5, 6], "fold": [0, 1], "each": [0, 1], "162": 0, "candid": [0, 1], "810": 0, "0001": 0, "243": 0, "1215": 0, "to_str": [0, 1, 4, 5, 6], "988": 0, "982": 0, "000": 0, "991": 0, "984": 0, "977": [0, 5, 6], "981": 0, "975": 0, "986": 0, "979": 0, "its": 0, "one": [0, 2], "much": 0, "faster": 0, "And": 0, "case": 0, "confusion_matrix": [0, 5, 6], "kei": [0, 1, 2, 5, 6], "y_pred": [0, 1, 4, 5, 6], "cm": [0, 5, 6], "d": [0, 5, 6], "roc_curv": [0, 5, 6], "fpr": [0, 5, 6], "tpr": [0, 5, 6], "plot": [0, 1, 2, 5, 6], "label": [0, 5, 6], "legend": [0, 5, 6], "mglearn": 1, "load_extended_boston": 1, "354": 1, "104": 1, "svr": 1, "model__epsilon": 1, "model_evaluation_lr": 1, "mape": 1, "rmse": 1, "r2": 1, "686": 1, "3430": 1, "278": 1, "1029": 1, "5145": 1, "001": [1, 5], "704": 1, "109": 1, "200": [1, 5, 6], "863": 1, "116": 1, "507": 1, "835": 1, "117": [1, 2, 4], "121": 1, "869": [1, 2], "129": [1, 2], "891": 1, "797": 1, "clear": 1, "winner": 1, "lowest": [1, 4], "execut": [1, 2], "realli": 1, "close": [1, 2], "fastest": 1, "lw": 1, "set_ylabel": [1, 2], "df_fraud": [2, 4], "data": [2, 4, 5, 6], "data_fraud": [2, 4], "transactionid": [2, 4], "isfraud": [2, 4, 5, 6], "transactiondt": [2, 4], "transactionamt": [2, 4], "productcd": [2, 4], "card1": [2, 4], "card2": [2, 4], "card3": [2, 4], "card4": [2, 4], "card5": [2, 4], "id_31": [2, 4], "id_32": [2, 4], "id_33": [2, 4], "id_34": [2, 4], "id_35": [2, 4], "id_36": [2, 4], "id_37": [2, 4], "id_38": [2, 4], "devicetyp": [2, 4], "deviceinfo": [2, 4], "2987000": [2, 4], "86400": [2, 4], "68": [2, 4], "w": [2, 4], "13926": [2, 4], "nan": [2, 4], "150": [2, 4], "discov": [2, 4], "142": [2, 4], "2987001": [2, 4], "86401": [2, 4], "29": [2, 4], "2755": [2, 4], "404": [2, 4], "mastercard": [2, 4], "102": [2, 4], "2987002": [2, 4], "86469": [2, 4], "59": [2, 4], "4663": [2, 4], "490": [2, 4], "visa": [2, 4], "166": [2, 4], "2987003": [2, 4], "86499": [2, 4], "18132": [2, 4], "567": [2, 4], "2987004": [2, 4], "86506": [2, 4], "h": [2, 4], "4497": [2, 4], "514": [2, 4], "samsung": [2, 4], "browser": [2, 4], "32": [2, 4], "2220x1080": [2, 4], "match_statu": [2, 4], "mobil": [2, 4], "sm": [2, 4], "g892a": [2, 4], "build": [2, 4], "nrd90m": [2, 4], "2987005": 2, "86510": 2, "49": 2, "5937": 2, "555": 2, "226": 2, "2987006": 2, "86522": 2, "159": 2, "12308": 2, "360": 2, "2987007": 2, "86529": 2, "422": 2, "12695": 2, "2987008": 2, "86535": 2, "2803": 2, "safari": 2, "1334x750": 2, "io": 2, "devic": 2, "2987009": 2, "86536": 2, "17399": 2, "111": 2, "224": [2, 6], "434": [2, 4], "print_percentage_missing_valu": 2, "115523073": 2, "preprocess": [2, 5, 6, 7], "ascend": [2, 4, 5, 6], "id_24": 2, "585793": 2, "99": 2, "id_25": 2, "585408": 2, "id_26": 2, "585377": 2, "id_21": 2, "585381": 2, "id_22": 2, "585371": 2, "id_23": 2, "id_08": 2, "585385": 2, "id_07": [2, 4], "id_27": 2, "dist2": 2, "552913": 2, "here": 2, "rate": 2, "590540": [2, 4], "590539": 2, "399": 2, "int64": 2, "gb": 2, "process": 2, "beacus": 2, "numer": 2, "an": 2, "encod": 2, "convert": 2, "colinear": 2, "numpi": [2, 4], "np": [2, 4], "df_numer": 2, "select_dtyp": [2, 4], "exclud": 2, "to_list": [2, 4], "addr1": 2, "addr2": 2, "dist1": 2, "c1": 2, "c2": 2, "c3": 2, "c4": 2, "c5": 2, "c6": 2, "c7": 2, "c8": 2, "c9": 2, "c10": 2, "c11": 2, "c12": 2, "c13": 2, "c14": 2, "d1": 2, "d2": [2, 4], "d3": 2, "d4": [2, 4], "d5": [2, 4], "d6": 2, "d7": 2, "d8": [2, 4], "d9": 2, "d10": [2, 4], "d11": 2, "d12": 2, "d13": 2, "d14": 2, "d15": [2, 4], "v1": 2, "v2": 2, "v3": 2, "v4": 2, "v5": 2, "v6": 2, "v7": 2, "v8": 2, "v9": 2, "v10": 2, "v11": 2, "v12": 2, "v13": 2, "v14": 2, "v15": 2, "v16": 2, "v17": 2, "v18": [2, 4], "v19": 2, "v20": 2, "v21": 2, "v22": 2, "v23": 2, "v24": 2, "v25": 2, "v26": 2, "v27": 2, "v28": 2, "v29": 2, "v30": [2, 4], "v31": 2, "v32": 2, "v33": 2, "v34": [2, 4], "v35": 2, "v36": [2, 4], "v37": 2, "v38": 2, "v39": 2, "v40": 2, "v41": 2, "v42": 2, "v43": 2, "v44": 2, "v45": 2, "v46": 2, "v47": 2, "v48": 2, "v49": 2, "v50": 2, "v51": 2, "v52": 2, "v53": 2, "v54": 2, "v55": 2, "v56": [2, 4], "v57": 2, "v58": 2, "v59": 2, "v60": 2, "v61": 2, "v62": [2, 4], "v63": 2, "v64": 2, "v65": 2, "v66": 2, "v67": 2, "v68": 2, "v69": 2, "v70": 2, "v71": 2, "v72": 2, "v73": 2, "v74": [2, 4], "v75": [2, 4], "v76": 2, "v77": 2, "v78": 2, "v79": 2, "v80": 2, "v81": 2, "v82": 2, "v83": 2, "v84": 2, "v85": 2, "v86": 2, "v87": [2, 4], "v88": 2, "v89": 2, "v90": 2, "v91": 2, "v92": 2, "v93": 2, "v94": 2, "v95": 2, "v96": 2, "v97": 2, "v98": 2, "v99": 2, "v100": 2, "v101": 2, "v102": 2, "v103": 2, "v104": 2, "v105": 2, "v106": 2, "v107": 2, "v108": 2, "v109": 2, "v110": 2, "v111": 2, "v112": 2, "v113": 2, "v114": 2, "v115": 2, "v116": 2, "v117": 2, "v118": 2, "v119": 2, "v120": 2, "v121": 2, "v122": 2, "v123": 2, "v124": 2, "v125": 2, "v126": 2, "v127": 2, "v128": 2, "v129": 2, "v130": 2, "v131": 2, "v132": 2, "v133": 2, "v134": 2, "v135": 2, "v136": 2, "v137": 2, "v138": 2, "v139": 2, "v140": 2, "v141": 2, "v142": [2, 4], "v143": 2, "v144": 2, "v145": [2, 4], "v146": 2, "v147": [2, 4], "v148": 2, "v149": 2, "v150": 2, "v151": 2, "v152": 2, "v153": 2, "v154": 2, "v155": 2, "v156": 2, "v157": 2, "v158": 2, "v159": 2, "v160": 2, "v161": 2, "v162": [2, 4], "v163": 2, "v164": 2, "v165": [2, 4], "v166": 2, "v167": 2, "v168": 2, "v169": [2, 4], "v170": 2, "v171": 2, "v172": 2, "v173": 2, "v174": 2, "v175": 2, "v176": [2, 4], "v177": 2, "v178": 2, "v179": 2, "v180": 2, "v181": 2, "v182": 2, "v183": 2, "v184": [2, 4], "v185": 2, "v186": 2, "v187": 2, "v188": 2, "v189": 2, "v190": 2, "v191": 2, "v192": 2, "v193": 2, "v194": 2, "v195": 2, "v196": 2, "v197": 2, "v198": 2, "v199": 2, "v200": 2, "v201": [2, 4], "v202": 2, "v203": 2, "v204": 2, "v205": 2, "v206": 2, "v207": 2, "v208": 2, "v209": 2, "v210": 2, "v211": 2, "v212": 2, "v213": 2, "v214": 2, "v215": 2, "v216": 2, "v217": 2, "v218": 2, "v219": 2, "v220": [2, 4], "v221": 2, "v222": [2, 4], "v223": 2, "v224": 2, "v225": 2, "v226": 2, "v227": 2, "v228": 2, "v229": 2, "v230": 2, "v231": 2, "v232": [2, 4], "v233": 2, "v234": 2, "v235": 2, "v236": 2, "v237": 2, "v238": 2, "v239": [2, 4], "v240": 2, "v241": 2, "v242": [2, 4], "v243": 2, "v244": [2, 4], "v245": 2, "v246": [2, 4], "v247": 2, "v248": 2, "v249": 2, "v250": 2, "v251": [2, 4], "v252": 2, "v253": 2, "v254": 2, "v255": 2, "v256": 2, "v257": [2, 4], "v258": 2, "v259": 2, "v260": 2, "v261": [2, 4], "v262": 2, "v263": 2, "v264": 2, "v265": 2, "v266": 2, "v267": 2, "v268": 2, "v269": 2, "v270": 2, "v271": 2, "v272": 2, "v273": 2, "v274": 2, "v275": 2, "v276": 2, "v277": 2, "v278": 2, "v279": 2, "v280": 2, "v281": [2, 4], "v282": [2, 4], "v283": [2, 4], "v284": 2, "v285": 2, "v286": 2, "v287": 2, "v288": 2, "v289": 2, "v290": 2, "v291": 2, "v292": 2, "v293": 2, "v294": 2, "v295": 2, "v296": 2, "v297": 2, "v298": 2, "v299": 2, "v300": 2, "v301": 2, "v302": 2, "v303": [2, 4], "v304": 2, "v305": 2, "v306": 2, "v307": 2, "v308": 2, "v309": 2, "v310": 2, "v311": 2, "v312": 2, "v313": 2, "v314": 2, "v315": 2, "v316": 2, "v317": 2, "v318": 2, "v319": 2, "v320": 2, "v321": 2, "v322": 2, "v323": 2, "v324": 2, "v325": 2, "v326": 2, "v327": 2, "v328": 2, "v329": 2, "v330": 2, "v331": 2, "v332": 2, "v333": 2, "v334": 2, "v335": 2, "v336": 2, "v337": 2, "v338": 2, "v339": 2, "id_01": [2, 4], "id_02": 2, "id_03": 2, "id_04": [2, 4], "id_05": 2, "id_06": 2, "id_09": 2, "id_10": 2, "id_11": 2, "id_13": 2, "id_14": 2, "id_17": 2, "id_18": 2, "id_19": 2, "id_20": 2, "df_card": 2, "startswith": 2, "corr_matrix_card": 2, "df_c": 2, "corr_matrix_c": 2, "df_d": 2, "corr_matrix_d": 2, "df_id": 2, "corr_matrix_id": 2, "somm": 2, "df_v_0_to_50": [], "corr_matrix_v": 2, "onc": 2, "nb": 2, "581607": 2, "588975": 2, "586281": 2, "9898": 2, "734658": 2, "362": 2, "555488": 2, "153": 2, "194925": 2, "199": 2, "278897": 2, "4901": 2, "170153": 2, "157": 2, "793246": 2, "336444": 2, "244453": 2, "6019": 2, "214": 2, "9678": 2, "361": 2, "14184": 2, "512": 2, "18396": 2, "600": 2, "231": 2, "237": 2, "seem": 2, "bimod": 2, "185": 2, "whearea": 2, "discret": 2, "0x7e01c5dbb820": [], "betwenn": 2, "let": 4, "random": [2, 5, 6], "reindex": 2, "colum": 2, "eda": 2, "092458": 2, "269734": 2, "005644": 2, "071082": 2, "480240": 2, "240343": 2, "076227": 2, "539918": 2, "569018": 2, "154": 2, "668899": 2, "150536": 2, "71": 2, "508467": 2, "674897": 2, "581443": 2, "86": 2, "666218": 2, "364844": 2, "4685": 2, "5691": 2, "26": 2, "2253": 2, "210": 2, "3257": 2, "3188": 2, "2918": 2, "unbalanc": 2, "highli": 2, "skew": 2, "0x7dff8a71f340": [], "alreadi": 2, "seen": 2, "befor": 2, "589271": 2, "309743": 2, "327662": 2, "73187": 2, "74926": 2, "514518": 2, "64717": 2, "62187": 2, "347568": 2, "169": 2, "563231": 2, "28": 2, "343348": 2, "805717": 2, "561057": 2, "123": 2, "982137": 2, "54": 2, "037533": 2, "724444": 2, "660387": 2, "177": 2, "315865": 2, "384721": 2, "143": 2, "669253": 2, "316880": 2, "615225": 2, "124": 2, "274558": 2, "136": 2, "312450": 2, "83": 2, "193": 2, "208333": 2, "666666": 2, "276": 2, "833333": 2, "640": 2, "819": 2, "873": 2, "958333": 2, "876": 2, "648": 2, "878": 2, "distinct": 2, "differ": 2, "0x7dff7b29ac70": [], "144233": 2, "140872": 2, "80044": 2, "139369": 2, "139318": 2, "5159": 2, "170502": 2, "174716": 2, "584708": 2, "091023": 2, "301124": 2, "344": 2, "507146": 2, "189": 2, "451377": 2, "353": 2, "128174": 2, "368": 2, "269820": 2, "347949": 2, "159651": 2, "816856": 2, "983842": 2, "789446": 2, "695502": 2, "375360": 2, "095343": 2, "198": 2, "847038": 2, "36": 2, "660": 2, "67992": 2, "266": 2, "252": 2, "125800": 2, "341": 2, "228749": 2, "225": 2, "427": 2, "486": 2, "999595": 2, "720": 2, "229": 2, "671": 2, "854": [2, 6], "0x7dff6cc49c70": [], "339": 2, "421571": 2, "590226": 2, "139631": 2, "130430": 2, "590528": 2, "168942": 2, "009298": 2, "000391": 2, "336": 2, "611559": 2, "856243": 2, "005597": 2, "123061": 2, "167660": 2, "433359": 2, "110179": 2, "035238": 2, "4238": 2, "666949": 2, "792934": 2, "460253": 2, "21": [2, 5], "021950": 2, "282454": 2, "107": 2, "949997": 2, "160000": 2, "337": 2, "880": 2, "0x7dff47e54760": [], "patern": 2, "alwai": 2, "sever": 2, "breast": 7, "cancer": 7, "boston": 7, "hous": 7, "fraud": 7, "df_object": 2, "includ": [2, 4], "nuniqu": 2, "1786": [], "260": 2, "id_30": 2, "r_emaildomain": 2, "p_emaildomain": 2, "card6": 2, "df_object1": 2, "df_object2": 2, "value_count": 2, "kind": 2, "bar": 2, "id23": 2, "id27": 2, "id28": 2, "etc": 2, "These": 2, "actual": 2, "creat": [], "missing_values_count": 2, "missing_values_percentag": 2, "missing_values_stat": 2, "522237": 2, "340": 2, "78": [2, 5], "522280": 2, "77": 2, "31095": 2, "76": 2, "474324": 2, "328": 2, "456111": 2, "327": 2, "456105": 2, "325": 2, "474339": 2, "474295": 2, "322": 2, "474270": 2, "456132": 2, "bin": 2, "xlabel": 2, "ylabel": 2, "signific": 2, "amount": 2, "understand": 2, "impact": 2, "fraudul": 2, "transact": 2, "similar": 2, "suggest": 2, "relat": 2, "map": 2, "less": 2, "interest": 2, "improv": 2, "df_v_50_featur": 2, "0x72d51a181ca0": 2, "092185": 2, "571526": 2, "848478": 2, "144574": 2, "241521": 2, "295215": 2, "848459": 2, "786976": 2, "727304": 2, "378574": 2, "336292": 2, "544262": 2, "349": 2, "2255": 2, "3331": 2, "1429": 2, "0x72d421a94040": 2, "421618": 2, "280699": 2, "38917": 2, "311253": 2, "61952": 2, "501427": 2, "140": 2, "002441": 2, "335965": 2, "638950": 2, "146": 2, "058108": 2, "621465": 2, "901295": 2, "163": 2, "744579": 2, "191": 2, "096774": 2, "89": 2, "000144": 2, "743264": 2, "663840": 2, "186": 2, "042622": 2, "614425": 2, "202": 2, "726660": 2, "53": [2, 4, 5, 6], "875000": 2, "253": 2, "187": 2, "958328": 2, "274": 2, "314": 2, "843": 2, "1707": 2, "791626": 2, "670": [2, 6], "847": 2, "879": 2, "0x72d2c31d1070": 2, "varibal": 2, "66324": 2, "136865": 2, "127320": 2, "5169": 2, "5132": 2, "060189": 2, "058938": 2, "615585": 2, "698710": 2, "48": 2, "053071": 2, "002708": 2, "329": 2, "608924": 2, "598231": 2, "701015": 2, "249856": 2, "491104": 2, "774858": 2, "897665": 2, "461089": 2, "321": 2, "371": 2, "64": [2, 4], "548": 2, "0x72d2b91ff670": 2, "assumpt": 2, "made": 2, "linearli": 2, "514467": 2, "81945": 2, "141416": 2, "82351": 2, "463915": 2, "988040": 2, "007739": 2, "000729": 2, "050415": 2, "777485": 2, "359005": 2, "013279": 2, "525103": 2, "087783": 2, "851040": 2, "352422": 2, "521522": 2, "209302": 2, "097290": 2, "036392": 2, "280037": 2, "647209": 2, "913772": 2, "625455": 2, "644": 2, "878586": 2, "512748": 2, "950295": 2, "668": 2, "486833": 2, "384": 2, "55125": 2, "104060": 2, "1st": 2, "3rd": 2, "quartil": 2, "must": 2, "unimod": 2, "again": 2, "obtain": 2, "zero": 2, "larg": 2, "part": 2, "descript": 2, "statist": 2, "0x72d27c20bb80": 2, "crash": 2, "code": 2, "current": 2, "cell": 2, "previou": 2, "pleas": 2, "review": 2, "identifi": 2, "possibl": 2, "caus": 2, "failur": 2, "click": 2, "href": 2, "http": [2, 4, 5, 6], "aka": 2, "vscodejupyterkernelcrash": 2, "view": 2, "jupyt": 2, "command": 2, "viewoutput": 2, "log": 2, "further": 2, "detail": 2, "sklearnex": [4, 5, 6], "patch_sklearn": [4, 5, 6], "intel": [4, 5, 6], "extens": [4, 5, 6], "scikit": [4, 5, 6], "learn": [4, 5, 6], "enabl": [4, 5, 6], "github": [4, 5, 6], "com": [4, 5, 6], "intelex": [4, 5, 6], "df_train": 4, "df_test": 4, "onehotencod": 4, "o": [4, 5, 6], "pickl": [4, 5, 6], "path_data_train_encod": 4, "encoded_train": 4, "pkl": [4, 5, 6], "path_data_test_encod": 4, "encoded_test": 4, "exist": [4, 5, 6], "object_df_train": 4, "object_df_test": 4, "handle_unknown": 4, "categori": 4, "categories_": 4, "encoded_column": 4, "extend": 4, "encoded_df_train": 4, "toarrai": 4, "encoded_df_test": 4, "df_train_non_categor": 4, "df_test_non_categor": 4, "reset_index": 4, "dump": [4, 5, 6], "open": [4, 5, 6], "wb": [4, 5, 6], "load": [4, 5, 6], "rb": [4, 5, 6], "472432": [4, 5, 6], "2718": 4, "118108": [4, 5, 6], "corrwith": 4, "387404": 4, "371294": 4, "368243": 4, "364785": 4, "332932": 4, "quit": 4, "elimin": 4, "threshold": 4, "arang": 4, "005": 4, "nb_col_remain": 4, "columns_to_drop": 4, "append": 4, "marker": 4, "remain": 4, "depend": 4, "If": 4, "lower": 4, "05": 4, "still": 4, "But": 4, "path_data_train_filt": 4, "filtered_train": 4, "path_data_test_filt": 4, "filtered_test": 4, "df_train_filt": 4, "df_test_filt": 4, "213": 4, "ensembl": [4, 5, 6], "randomforestclassifi": [4, 5, 6], "experiment": 4, "enable_iterative_imput": 4, "iterativeimput": 4, "path_results_imput": 4, "results_imput": 4, "max_it": 4, "iter": 4, "x_train_imput": 4, "x_test_imput": 4, "y_train_imput": 4, "y_test_imput": 4, "n_nearest_featur": 4, "n_estim": 4, "max_depth": 4, "path_data_train_imput": 4, "imputed_train": 4, "path_data_test_imput": 4, "imputed_test": 4, "df_fraud_train_imput": 4, "df_fraud_test_imput": 4, "df_fraud_imput": 4, "delete_multicollinear": 4, "path_data_train_preprocess": 4, "preprocessed_train": [4, 5, 6], "path_data_test_preprocess": 4, "preprocessed_test": [4, 5, 6], "df_sampl": 4, "frac": 4, "df_fraud_train_preprocess": 4, "columns_to_keep": 4, "list": [4, 6], "df_fraud_train_fin": 4, "df_fraud_test_fin": 4, "card6_credit": 4, "r_emaildomain_gmail": 4, "m2_nan": 4, "m3_t": 4, "m4_m2": 4, "m4_nan": 4, "m6_t": 4, "m6_nan": 4, "m9_t": 4, "id_12_notfound": 4, "id_31_chrom": 4, "android": 4, "gener": 4, "id_37_t": 4, "devicetype_mobil": 4, "deviceinfo_sm": 4, "a300h": 4, "lrx22g": 4, "deviceinfo_hi6210sft": 4, "mra58k": 4, "path_train": [5, 6], "path_test": [5, 6], "data_train": [5, 6], "data_test": [5, 6], "16421": [5, 6], "32842": [5, 6], "counter": [5, 6], "imblearn": [5, 6], "over_sampl": [5, 6], "adasyn": [5, 6], "ada": [5, 6], "x_train_r": [5, 6], "y_train_r": [5, 6], "fit_resampl": [5, 6], "origin": [5, 6], "resampl": [5, 6], "113866": [5, 6], "4242": [5, 6], "skopt": 5, "space": 5, "real": 5, "categor": 5, "integ": 5, "forest": [5, 6], "model__max_depth": [5, 6], "model__n_estim": [5, 6], "bayessearchcv": 5, "model_path": [5, 6], "model_tim": [5, 6], "cpu_tim": [5, 6], "_time": [5, 6], "search_spac": 5, "n_iter": 5, "durat": [5, 6], "ordereddict": 5, "62878302619048": 5, "04745226486541268": 5, "2067": 5, "963": [5, 6], "496": 5, "992": 5, "661": 5, "1112": 5, "495": 5, "951": 5, "651": 5, "957": [5, 6], "828": 5, "807": 5, "711": 5, "209": 5, "760": 5, "best_estimator_": [5, 6], "named_step": [5, 6], "feature_importances_": [5, 6], "indic": [5, 6], "argsort": [5, 6], "barplot": [5, 6], "1314": 6, "965": 6, "506": 6, "990": 6, "1032": 6, "492": 6, "952": 6, "649": 6, "958": 6, "14512": 6, "964": 6, "501": 6, "949": 6, "656": 6, "36147": 6, "940": 6, "367": 6, "924": 6, "525": 6, "932": 6, "837": 6, "750": 6, "bay": 7, "optim": 7}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"welcom": [], "your": [], "jupyt": [], "book": [], "exercis": [0, 1, 2, 3, 4, 5, 6], "2": [2, 3], "wind": 3, "speed": 3, "clean": 3, "data": [0, 1, 3], "eda": 3, "1": [0, 1], "breast": 0, "cancer": 0, "load": [0, 1], "preprocess": [0, 1, 4], "train": [0, 1, 5, 6], "result": [0, 1], "boston": 1, "hous": 1, "fraud": [2, 4, 5, 6], "detect": [2, 4, 5, 6], "card": 2, "featur": 2, "c": 2, "d": 2, "id": 2, "v": 2, "partial": 7, "3": [4, 5, 6, 7], "c\u00e9lien": 7, "bonhomm": 7, "leonardo": 7, "vaia": 7, "dataset": [], "split": 4, "encod": 4, "imput": 4, "evalu": 4, "delet": 4, "multicolinear": 4, "vif": 4, "undersampl": [5, 6], "oversampl": [5, 6], "bay": 5, "optim": 5}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 60}, "alltitles": {"Exercise 1: Breast cancer": [[0, "exercise-1-breast-cancer"]], "Load data": [[0, "load-data"], [1, "load-data"]], "Preprocessing": [[0, "preprocessing"], [1, "preprocessing"]], "Training": [[0, "training"], [1, "training"], [5, "training"], [6, "training"]], "Results": [[0, "results"], [1, "results"]], "Exercise 1: Boston Housing": [[1, "exercise-1-boston-housing"]], "Exercise 2: Fraud detection": [[2, "exercise-2-fraud-detection"]], "card features": [[2, "card-features"]], "C features": [[2, "c-features"]], "D features": [[2, "d-features"]], "id features": [[2, "id-features"]], "V features": [[2, "v-features"]], "Exercise 2: Wind speed": [[3, "exercise-2-wind-speed"]], "Clean data": [[3, "clean-data"]], "EDA": [[3, "eda"]], "Partial 3": [[7, "partial-3"]], "C\u00e9lien BONHOMME - Leonardo VAIA": [[7, "celien-bonhomme-leonardo-vaia"]], "Exercise 3: Fraud detection preprocessing": [[4, "exercise-3-fraud-detection-preprocessing"]], "Splitting": [[4, "splitting"]], "Encoding": [[4, "encoding"]], "Imputation": [[4, "imputation"]], "Evaluation of the imputation": [[4, "evaluation-of-the-imputation"]], "Delete multicolinearity with VIF": [[4, "delete-multicolinearity-with-vif"]], "Exercise 3: Fraud detection training (Bayes Optimization)": [[5, "exercise-3-fraud-detection-training-bayes-optimization"]], "Undersampling": [[5, "undersampling"], [6, "undersampling"]], "Oversampling": [[5, "oversampling"], [6, "oversampling"]], "Exercise 3: Fraud detection training": [[6, "exercise-3-fraud-detection-training"]]}, "indexentries": {}})