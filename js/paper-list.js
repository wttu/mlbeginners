var papers = new Array (

    {
        id : "thesis_hku",
        name : "User-generated Content based Recommendation Systems for Investment and E-commerce.",
        description : "Postgraduate Thesis, 收录于The HKU Scholars Hub（香港大学学术库）",
        conference : "Dissertation",
        type : "d",
		year : "",
        paperAbstract : "<p>Nowadays, user-generated content based recommendation systems (UGC-Recsys) have become a very popular trend in recent years. In this thesis, recommendation systems based on three types of user-generated data are proposed, and their efficient evaluation is studied. First, this thesis focuses on a very useful user-generated content in investment domain, i.e., online investor opinions. The posts of investors who publish their investment views and suggestions are very useful for users who make investing decisions according to them. However, these online investor opinions vary greatly in quality and are of large scales. Thus, this thesis studies how to extract high-quality investor opinions from a massive collection of investor opinions posted on the web and make use of them for improving investment recommendation tasks. Our experiments on real-world datasets will show that the approaches presented in this thesis could recommend investor opinions and stocks that would bring users substantial profits. Our attention is then turned to product reviews that contain users’ viewpoints about products. Our goal is to improve recommendation of product reviews, by using personalization criteria. This is motivated by the fact that the importance of product aspects to different users may vary and users prefer to focus on the most important aspects to them. Previous work on ranking reviews based on quality and coverage is improved in the thesis to also consider the personalized preferences of users on product aspects. An experimental evaluation with two public review datasets demonstrates the effectiveness of our approach on recommending reviews that have high quality, coverage, and relevance to the aspects that are important for the user. A wide range of user-generated content exists in the e-commerce domain. Product reviews are one kind of unstructured user-generated content while there are also several kinds of more structured user-generated content. For example, peoples could share their friend lists (social connections) or their check-in records (geographical locations) in many location based social network (LBSN) websites such as Foursquare and Gowalla. This thesis focuses on social connections and check-in records obtained from LBSN platforms. For convenience, we call the collection of social connections and check-in records from LBSN platforms as LBSN data. Based on LBSN data, location recommendation becomes a typical function of location service based e-commerce sites. In this thesis, we focus on the social-activity related locations (briefly denoted as activity locations) which indicate the locations could be mapped to real-world social activities. Our work is the first one to consider an important characteristic of activity locations: people like to go to these activity locations with their friends together, and improves the effectiveness of recommendation on activity locations in two directions. First, the problem of location-partner recommendation (i.e., for each recommended activity location, find a suitable partner for the user) is studied. Second, assuming that users tend to select the activities for which they can find suitable partners, a partner-aware activity location recommendation model is proposed. Finally, experiments on real data are conducted, which verify the effectiveness of location-partner recommendation and partner aware activity-location recommendation.</p>",
    },

    {
        id : "thesis_ecnu",
        name : "域适应算法以及基于用户迁移的个性化机器学习应用",
        description : "硕士学位论文，收录于华东师范大学图书馆，被评为<font style='color: rgb(229, 51, 51)'>上海市研究生优秀成果（学位论文）</font>",
        conference : "Dissertation",
        type : "d",
		year : "",
        paperAbstract : "<p>本文专注于域适应算法上的研究,并探索域适应算法在基于用户迁移的个性化机器学习上的应用潜力。在理论方面,本文提出了三种创新的域适应算法。另外本文在实际应用方法特别提出了利用域适应算法来实现基于用户迁移的机器学习应用,并以脑电信号分类系统为例展现了这一思想的重要性和可行性。 域适应问题在机器学习领域已获得了越来越广泛的关注。它对应于训练样本和测试样本是基于不同分布的情景,这在真实的应用上是非常普遍的。例如对于面向用户的服务型机器学习应用(例如脑电分类,语音识别,脸部识别)上,常常为了提高用户体验以及一些实际的约束而面临训练样本和测试样本是来自于不同的用户群体,此时由于用户群体上的分类模式差异,会导致训练模型不适合目标用户。又例如自然语言处理中,也常常要面临训练预料和测试样例的数据分布差异。所以,域适应是一个十分具有研究价值的问题。本文提出了三种创新的域适应算法,从特征表示和分类模型阶段都对传统方法做出了改进,并以脑电信号分类为例探讨它们在实现基于用户迁移的机器学习应用上的前景。 本文提出的第一个域适应的算法是可迁移的判别式特征降维方法,它从判别性和迁移性两个方面来同时优化低维空间。从而避免了传统的判别性降维方法得到的低维空间将过拟合于源数据的危险。它通过设计度量域适应情景中低维空间的判别性和迁移性的数学项,然后通过优化由它们构成的目标函数来学习到低维空间。从而使得在低维空间中,不仅保持了数据的判别性,也能够增加源域模型对于目标域的迁移性。本文通过利用模拟数据和真实数据上的实验来验证了算法能够学习到一个更加适合域适应情景的低维空间。并通过可视化这些低维空间来更加验证了它们的优势。 本文提出的第二个算法是基于稳健适应双视角和反向近邻策略的动态域适应,它的思想是利用集成学习框架来结合多个源用户数据产生的基本模型。算法在两个层次上提出了创新,在构建基本模型集合时,从适应性和稳健性两方面来进行优化从而构建具有互补性的基本模型集。适应性是针对于适应目标域而言,稳健性是针对于能够稳健于域之间差异而言。另外,在决定模型权值层次上,算法利用了基于反向近邻策略的动态加权准则。利用每个测试样本在源数据上的近邻们的结构来动态决定权值。在九个真实数据上的实验表明了算法的有效性。 本文提出的最后一个算法是针对于消除对目标域样本限制的出发点设计的域适应算法。通过利用集成学习框架将动态分配权值转化为可以利用统计分类理论的二类分类问题,算法为每个基本模型建立了一个模型友好分类器,这个分类器的训练目标是为了预测一个样本是否适合于对应模型来进行后续任务。如此,针对于每个测试样本,能够根据基本模型集的模型友好分类器集的结果来决定最后在组合分类器中的权值。算法不要求目标域具有训练或测试样本,只是根据源域和目标域的分布不同的事实来更加小心地决定权值,从而增加模型对于测试样本的泛化性能。在真实数据集上的实验结果证明了算法的有效性。 总结下来,本文不仅从理论上提出了域适应领域的创新算法,也结合探讨了有前景的应用方向。从理论和应用两个层次上都具有一定的创新价值。</p>",
    },	
	

    {
        id : "irec_IS",
        name : "Investment Recommendation by Discovering High-quality Opinions in Investor based Social Networks.",
		tags : new Array(tagList.ml, tagList.recsys, tagList.fintech),
        coauthors : new Array(authorList.wttu, authorList.myang, authorList.david, authorList.nikos),
        conference : "Information Systems (Elsevier)",
        type : "j",
		year : "2018",
        paperAbstract : "<p>Investor based social networks, such as StockTwist, are gaining increasing popularity. These sites allow users to post their investment opinions in the form of microblogs. Given the growth of the posted data, a significant and challenging research problem is how to utilize the personal wisdom and different viewpoints in these opinions to help investment. A typical way is to aggregate sentiments related to stocks and generates buy or hold recommendations for stocks obtaining favorable votes while suggesting sell or short actions for stocks with negative votes. However, considering the fact that there always exist unreasonable or misleading posts, sentiment aggregation should be improved to be robust to noise. In our work, we study how to estimate qualities of investment opinions in investor based social networks. To predict the quality of an investment opinion, we use multiple categories of factors generated from the author information, opinion content and the characteristics of stocks to which the opinion refers. With predicted qualities of investment opinions, we perform two types of investment recommendation. The first is recommending high-quality opinions to users and the second is recommending portfolios generated by sentiment aggregation in a quality-sensitive manner. Experimental results on real datasets demonstrate the effectiveness of our work in recommending high-quality investment opinions and profitable portfolios.</p>",
    },


  {
        id : "pds_NN",
        name : "Personalized Response Generation by Dual-learning based Domain Adaptation.",
		tags : new Array(tagList.dl, tagList.nlp),
        coauthors : new Array(authorList.myang, authorList.wttu, authorList.qqiang, authorList.zzhao, authorList.xjchen, authorList.jzhu),
        conference : "Neural Networs",
        type : "j",
		year : "2018",
        paperAbstract : "<p>Open-domain conversation is one of the most challenging artificial intelligence problems, which involves language understanding, reasoning, and the utilization of common sense knowledge. The goal of this paper is to further improve the response generation, using personalization criteria. We propose a novel method called PRGDDA (Personalized Response Generation by Dual-learning based Domain Adaptation) which is a personalized response generation model based on theories of domain adaptation and dual learning. During the training procedure, PRGDDA first learns the human responding style from large general data (without user-specific information), and then fine-tunes the model on a small size of personalized data to generate personalized conversations with a dual learning mechanism. We conduct experiments to verify the effectiveness of the proposed model on two real-world datasets in both English and Chinese. Experimental results show that our model can generate better personalized responses for different users.",
    },	
	
    {
        id : "apr_tweb",
        name : "Activity Recommendation with Partners.",
		tags : new Array(tagList.recsys),
        coauthors : new Array(authorList.wttu, authorList.david, authorList.nikos, authorList.myang, authorList.zylu),
        conference : "ACM Transactions on the Web (TWEB)",
        type : "j",
		year : "2017",
        paperAbstract : "<p>Recommending social activities, such as watching movies or having dinner, is a common function found in social networks or e-commerce sites. Besides certain websites which manage activity-related locations (e.g., foursquare.com), many items on product sale platforms (e.g., groupon.com) can naturally be mapped to social activities. For example, movie tickets can be thought of as activity items, which can be mapped as a social activity of “watch a movie.” Traditional recommender systems estimate the degree of interest for a target user on candidate items (or activities), and accordingly, recommend the top-k activity items to the user. However, these systems ignore an important social characteristic of recommended activities: people usually tend to participate in those activities with friends. This article considers this fact for improving the effectiveness of recommendation in two directions. First, we study the problem of activity-partner recommendation; i.e., for each recommended activity item, find a suitable partner for the user. This (i) saves the user’s time for finding activity partners, (ii) increases the likelihood that the activity item will be selected by the user, and (iii) improves the effectiveness of recommender systems to users overall and enkindles their social enthusiasm. Our partner recommender is built upon the users’ historical attendance preferences, their social context, and geographic information. Moreover, we explore how to leverage the partner recommendation to help improve the effectiveness of recommending activities to users. Assuming that users tend to select the activities for which they can find suitable partners, we propose a partner-aware activity recommendation model, which integrates this hypothesis into conventional recommendation approaches. Finally, the recommended items not only match users’ interests, but also have high chances to be selected by the users, because the users can find suitable partners to attend the corresponding activities together. We conduct experiments on real data to evaluate the effectiveness of activity-partner recommendation and partner-aware activity recommendation. The results verify that (i) suggesting partners greatly improves the likelihood that a recommended activity item is to be selected by the target user and (ii) considering the existence of suitable partners in the ranking of recommended items improves the accuracy of recommendation significantly.</p>",
    },	

    {
        id : "ptrs_neurocomputing",
        name : "More Focus on What You Care About: Personalized Top Reviews Set.",
		tags : new Array(tagList.nlp, tagList.recsys),
        coauthors : new Array(authorList.wttu, authorList.david, authorList.nikos),
        conference : "Neurocomputing",
        type : "j",
		year : "2017",
        paperAbstract : "<p>Users of e-commerce sites often read reviews of products before deciding to purchase them. Many commercial sites simply select the reviews with the highest quality, according to the votes they have received by users who read the reviews. However, recent work has shown that such a selection may contain redundant information. Therefore, while selecting top reviews, it has been proposed to also consider their coverage (i.e., how many product aspects are covered by them). The goal of this paper is to further improve the top reviews set, using personalization criteria. This is motivated by the fact that the importance of product aspects to different users may vary and users prefer to focus on the most important aspects to them. The objective of our work is to consider the personal preferences of users in review recommendation, by selecting a personalized top reviews set (PTRS), which includes reviews of which the content is related to the aspects important to the user. An experimental evaluation with two public review datasets demonstrates the effectiveness of our approach on computing PTRS that have high quality, coverage, and relevance to the aspects that are important for the user.</p>",
    },
	
    {
        id : "trm_neurocomputing",
        name : "A Topic Drift Model for Authorship Attribution.",
		tags : new Array(tagList.nlp),
        coauthors : new Array(authorList.myang, authorList.xjchen, authorList.wttu, authorList.zylu, authorList.jzhu, authorList.qfin,),
        conference : "Neurocomputing",
        type : "j",
		year : "2017",
        paperAbstract : "<p>Authorship attribution is an active research direction due to its legal and financial importance. Its goal is to identify the authorship from the anonymous texts. In this paper, we propose a Topic Drift Model (TDM), which can monitor the dynamicity of authors’ writing styles and learn authors’ interests simultaneously. Unlike previous authorship attribution approaches, our model is sensitive to the temporal information and the ordering of words. Thus it can extract more information from texts. The experimental results show that our model achieves better results than other models in terms of accuracy. We also demonstrate the potential of our model to address the authorship verification problem.</p>",
    },
	
    {
        id : "plr_geoinf",
        name : "Personalized Location Recommendation by Aggregating Multiple Recommenders in Diversity.",
		tags : new Array(tagList.ml, tagList.recsys),
        coauthors : new Array(authorList.zylu, authorList.hwang, authorList.nikos, authorList.wttu, authorList.david),
        conference : "Geoinformatica",
        type : "j",
		year : "2017",
        paperAbstract : "<p>Location recommendation is an important feature of social network applications and location-based services. Most existing studies focus on developing one single method or model for all users. By analyzing data from two real location-based social networks (Foursquare and Gowalla), in this paper we reveal that the decisions of users on place visits depend on multiple factors, and different users may be affected differently by these factors. We design a location recommendation framework that combines results from various recommenders that consider different factors. Our framework estimates, for each individual user, the underlying influence of each factor to her. Based on the estimation, we aggregate suggestions from different recommenders to derive personalized recommendations. Experiments on Foursquare and Gowalla show that our proposed method outperforms the state-of-the-art methods on location recommendation.</p>",
    },

    {
        id : "cps_icdcs",
        name : "Detecting Time Synchronization Attacks in Cyber-Physical Systems with Machine Learning Techniques.",
		tags : new Array(tagList.ml, tagList.sec),
        coauthors : new Array(authorList.jxwang, authorList.wttu, authorList.hui, authorList.yiu),
        conference : "IEEE Proceedings of the 37th International Conference on Distributed Computing Systems (ICDCS)",
        type : "c",
		year : "2017",
        paperAbstract : "<p>Recently, researchers found a new type of attacks, called time synchronization attack (TS attack), in cyber-physical systems. Instead of modifying the measurements from the system, this attack only changes the time stamps of the measurements. Studies show that these attacks are realistic and practical. However, existing detection techniques, e.g. bad data detection (BDD) and machine learning methods, may not be able to catch these attacks. In this paper, we develop a ''first difference aware'' machine learning (FDML) classifier to detect this attack. The key concept behind our classifier is to use the feature of ''first difference'', borrowed from economics and statistics. Simulations on IEEE 14-bus system with real data from NYISO have shown that our FDML classifier can effectively detect both TS attacks and other cyber attacks.</p>",
    },	

	
    {
        id : "alstm_aaai",
        name : "Attention-based LSTM for Target-dependent Sentiment Classification.",
		tags : new Array(tagList.nlp, tagList.dl),
        coauthors : new Array(authorList.myang, authorList.wttu, authorList.jxwang, authorList.fxu, authorList.xjchen),
        conference : "Proceedings of the 31th AAAI Conference on Artificial Intelligence (AAAI)",
        type : "c",
		year : "2017",
        paperAbstract : "<p>We present an attention-based bidirectional LSTM approach to improve the target-dependent sentiment classification. Our method learns the alignment between the target entities and the most distinguishing features. We conduct extensive experiments on a real-life dataset. The experimental results show that our model achieves state-of-the-art results.</p>",
    },	
	
    {
        id : "irec_sigir",
        name : "Investment Recommendation using Investor Opinions in Social Media.",
		tags : new Array(tagList.ml, tagList.recsys, tagList.fintech),
        coauthors : new Array(authorList.wttu, authorList.david, authorList.nikos, authorList.myang, authorList.zylu),
        conference : "Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)",
        type : "c",
		year : "2016",
        paperAbstract : "<p>Investor social media, such as StockTwist, are gaining increasing popularity. These sites allow users to post their investing opinions and suggestions in the form of microblogs. Given the growth of the posted data, a significant and challenging research problem is how to utilize the personal wisdom and different viewpoints in these opinions to help investment. Previous work aggregates sentiments related to stocks and generates buy or hold recommendations for stocks obtaining favorable votes while suggesting sell or short actions for stocks with negative votes. However, considering the fact that there always exist unreasonable or misleading posts, sentiment aggregation should be improved to be robust to noise. In this paper, we improve investment recommendation by modeling and using the quality of each investment opinion. To model the quality of an opinion, we use multiple categories of features generated from the author information, opinion content and the characteristics of stocks to which the opinion refers. Then, we discuss how to perform investment recommendation (including opinion recommendation and portfolio recommendation) with predicted qualities of investor opinions. Experimental results on real datasets demonstrate effectiveness of our work in recommending high-quality opinions and generating profitable investment decisions.</p>",
    },	
	
    {
        id : "dai_sigir",
        name : "Discovering Author Interest Evolution in Topic Modeling.",
		tags : new Array(tagList.nlp, tagList.recsys),
        coauthors : new Array(authorList.myang, authorList.jcmei, authorList.fxu, authorList.wttu, authorList.zylu),
        conference : "Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)",
        type : "c",
		year : "2016",
        paperAbstract : "<p>Discovering the author's interest over time from documents has important applications in recommendation systems, authorship identification and opinion extraction. In this paper, we propose an interest drift model (IDM), which monitors the evolution of author interests in time-stamped documents. The model further uses the discovered author interest information to help finding better topics. Unlike traditional topic models, our model is sensitive to the ordering of words, thus it extracts more information from the semantic meaning of the context. The experiment results show that the IDM model learns better topics than state-of-the-art topic models.</p>",
    },	

	
    {
        id : "tsom_aaai",
        name : "Time-sensitive Opinion Mining for Prediction. ",
		tags : new Array(tagList.nlp, tagList.fintech),
        coauthors : new Array(authorList.wttu, authorList.david, authorList.nikos),
        conference : "Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)",
        type : "c",
		year : "2015",
        paperAbstract : "<p>Users commonly use Web 2.0 platforms to post their opinions and their predictions about future events (e.g., the movement of astock). Therefore, opinion mining can be used as a tool for predicting future events. Previous work on opinion mining extracts from the text only the polarity of opinions as sentiment indicators. We observe that a typical opinion post also contains temporal references which can improve prediction. This short paper presents our preliminary work on extracting reference time tagsand integrating them into an opinion mining model, in order to improvethe accuracy of future event prediction. We conduct anexperimental evaluation using a collection of microblogs posted by investors to demonstrate the effectiveness of our approach.</p>",
    },	
	
    {
        id : "autocorpus_aaai",
        name : "Improving Microblog Rtrieval from Exterior Corpus by Automatically Constructing Microblogging Corpus.",
		tags : new Array(tagList.nlp),
        coauthors : new Array(authorList.wttu, authorList.david, authorList.nikos),
        conference : "Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)",
        type : "c",
		year : "2015",
        paperAbstract : "<p>A large-scale training corpus consisting of microblogs belonging to a desired category is important for high-accuracy microblog retrieval. Obtaining such a large-scale microblgging corpus manually is very time and labor-consuming. Therefore, some models for the automatic retrieval of microblogs froman exterior corpus have been proposed. However, these approaches may fail in considering microblog-specific features. To alleviate this issue, we propose a methodology that constructs a simulated microblogging corpus rather than directly building a model from the exterior corpus. The performance of our model is better since the microblog-special knowledge of the microblogging corpus is used in the end by the retrieval model. Experimental results on real-world microblogs demonstrate the superiority of our technique compared to the previous approaches.</p>",
    },	
	
    {
        id : "ostopic_aaai",
        name : "Order-sensitive and Semantic-aware Topic Modeling Microblogging Corpus. ",
		tags : new Array(tagList.nlp),
        coauthors : new Array(authorList.myang, authorList.tycui, authorList.wttu),
        conference : "Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)",
        type : "c",
		year : "2015",
        paperAbstract : "<p>Topic modeling of textual corpora is an important and challenging problem. In most previous work, the “bag-of-words” assumption is usually made which ignores the ordering of words. This assumption simplifies the computation, but it unrealistically loses the ordering information and the semantic of words in the context. In this paper, we present a Gaussian Mixture Neural Topic Model (GMNTM) which incorporates both the ordering of words and the semantic meaning of sentences into topic modeling. Specifically, we represent each topic as a cluster of multi-dimensional vectors and embed the corpus into a collection of vectors generated by the Gaussian mixture model. Each word is affected not only by its topic, but also by the embedding vector of its surrounding words and the context. The Gaussian mixture components and the topic of documents, sentences and words can be learnt jointly. Extensive experiments show that our model can learn better topics and more accurate word distributions for each topic. Quantitatively, comparing to state-of-the-art topic modeling approaches, GMNTM obtains significantly better performance in terms of perplexity, retrieval accuracy and classification accuracy.</p>",
    },	

		
    {
        id : "apr_pakdd",
        name : "Activity Partner Recommendation.",
		tags : new Array(tagList.recsys),
        coauthors : new Array(authorList.wttu, authorList.david, authorList.nikos, authorList.myang, authorList.zylu),
        conference : "Proceedings of the 19th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)",
        type : "c",
		year : "2015",
        paperAbstract : "<p>In many activities, such as watching movies or having dinner, people prefer to find partners before participation. Therefore, when recommending activity items (e.g., movie tickets) to users, it makes sense to also recommend suitable activity partners. This way, (i) the users save time for finding activity partners, (ii) the effectiveness of the item recommendation is increased (users may prefer activity items more if they can find suitable activity partners), (iii) recommender systems become more interesting and enkindle users’ social enthusiasm. In this paper, we identify the usefulness of suggesting activity partners together with items in recommender systems. In addition, we propose and compare several methods for activity-partner recommendation. Our study includes experiments that test the practical value of activity-partner recommendation and evaluate the effectiveness of all suggested methods as well as some alternative strategies.</p>",
    },	
		
    {
        id : "realnews_paclic",
        name : "Real-time Detection and Sorting of News on Microblogging Platforms. ",
		tags : new Array(tagList.nlp),
        coauthors : new Array(authorList.wttu, authorList.david, authorList.nikos, authorList.myang, authorList.zylu),
        conference : "Proceedings of the 29th Pacific Asia Conference on Language, Information and Computing (PACLIC)",
        type : "c",
		year : "2015",
        paperAbstract : "<p>Due to the increasing popularity of microblogging platforms (e.g., Twitter), detecting real-time news from microblogs (e.g., tweets) has recently drawn much attention. Most of the previous work on this subject detect news by analyzing propagation patterns of microblogs. This approach has two limitations: (i) many non-news microblogs (e.g. marketing activities) have propagation patterns similar to news microblogs and therefore can be falsely reported as news; (ii) using propagation patterns to identify news involves a time delay until the pattern is formed, therefore news are not detected early. We propose an alternative approach, which, motivated by the necessity of early detection of news, does not rely on propagation of posts. Moreover, an early sorting strategy is also proposed to define an order of values of detected news microblogs using a translational approach. An experimental evaluation on a large-scale microblogging dataset demonstrates the effectiveness of our approach.</p>",
    },	
		
    {
        id : "dmnn_acl",
        name : "Deep Markov Neural Network for Sequential Data Classification.",
		tags : new Array(tagList.dl, tagList.nlp),
        coauthors : new Array(authorList.myang, authorList.wttu, authorList.wpyin, authorList.zylu),
        conference : "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics (ACL)",
        type : "c",
		year : "2015",
        paperAbstract : "<p>We present a general framework for incorporating sequential data and arbitrary features into language modeling. The general framework consists of two parts: a hidden Markov component and a recursive neural network component. We demonstrate the effectiveness of our model by applying it to a specific application: predicting topics and sentiments in dialogues. Experiments on real data demonstrate that our method is substantially more accurate than previous methods.</p>",
    },	

		
    {
        id : "sssa_naacl",
        name : "LCCT: A Semi-supervised Model for Sentiment Classification.",
		tags : new Array(tagList.nlp),
        coauthors : new Array(authorList.myang, authorList.wttu, authorList.zylu, authorList.chow),
        conference : "Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)",
        type : "c",
		year : "2015",
        paperAbstract : "<p>Analyzing public opinions towards products, services and social events is an important but challenging task. An accurate sentiment analyzer should take both lexicon-level information and corpus-level information into account. It also needs to exploit the domainspecific knowledge and utilize the common knowledge shared across domains. In addition, we want the algorithm being able to deal with missing labels and learning from incomplete sentiment lexicons. This paper presents a LCCT (Lexicon-based and Corpus-based, Co-Training) model for semi-supervised sentiment classification. The proposed method combines the idea of lexicon-based learning and corpus-based learning in a unified cotraining framework. It is capable of incorporating both domain-specific and domainindependent knowledge. Extensive experiments show that it achieves very competitive classification accuracy, even with a small portion of labeled data. Comparing to state-ofthe-art sentiment classification methods, the LCCT approach exhibits significantly better performances on a variety of datasets in both English and Chinese.</p>",	
    },	


    {
        id : "del_icpr",
        name : "Dynamical Ensemble Learning with Model-friendly Classifiers for Domain Adaptation. ",
		tags : new Array(tagList.ml),
        coauthors : new Array(authorList.wttu, authorList.slsun),
        conference : "Proceedings of the 21st International Conference on Pattern Recognition (ICPR)",
        type : "c",
		year : "2014b",
        paperAbstract : "<p>In the domain adaptation research, which recently becomes one of the most important research directions in machine learning, source and target domains are with different underlying distributions. In this paper, we propose an ensemble learning framework for domain adaptation. Owing to the distribution differences between source and target domains, the weights in the final model are sensitive to target examples. As a result, our method aims to dynamically assign weights to different test examples by making use of additional classifiers called model-friendly classifiers. The model-friendly classifiers can judge which base models predict well on a specific test example. Finally, the model can give the most favorable weights to different examples. In the experiments, we firstly testify the need of dynamical weights in the ensemble learning based domain adaptation, then compare our method with other classical methods on real datasets. The experimental results show that our method can learn a final model performing well in the target domain.</p>",
    },	


    {
        id : "ssbci_paa",
        name : "Semi-supervised Feature Extraction for EEG Classification. ",
		tags : new Array(tagList.ml, tagList.bci),
        coauthors : new Array(authorList.wttu, authorList.slsun),
        conference : "Pattern Analysis and Applications (PAA)",
        type : "j",
		year : "2014b",
        paperAbstract : "<p>Two semi-supervised feature extraction methods are proposed for electroencephalogram (EEG) classification. They aim to alleviate two important limitations in brain---computer interfaces (BCIs). One is on the requirement of small training sets owing to the need of short calibration sessions. The second is the time-varying property of signals, e.g., EEG signals recorded in the training and test sessions often exhibit different discriminant features. These limitations are common in current practical applications of BCI systems and often degrade the performance of traditional feature extraction algorithms. In this paper, we propose two strategies to obtain semi-supervised feature extractors by improving a previous feature extraction method extreme energy ratio (EER). The two methods are termed semi-supervised temporally smooth EER and semi-supervised importance weighted EER, respectively. The former constructs a regularization term on the preservation of the temporal manifold of test samples and adds this as a constraint to the learning of spatial filters. The latter defines two kinds of weights by exploiting the distribution information of test samples and assigns the weights to training data points and trials to improve the estimation of covariance matrices. Both of these two methods regularize the spatial filters to make them more robust and adaptive to the test sessions. Experimental results on data sets from nine subjects with comparisons to the previous EER demonstrate their better capability for classification.</p>",
    },	


    {
        id : "cdrl_sigkddw",
        name : "Cross-domain representation-learning framework with combination of class-separate and domain-merge objectives. ",
		tags : new Array(tagList.ml),
        coauthors : new Array(authorList.wttu, authorList.slsun),
        conference : "Workshop on Cross Domain Knowledge Discovery in Web and Social Network Mining (KDD-Workshop)",
        type : "c",
		year : "2014b",
        paperAbstract : "<p>Recently, cross-domain learning has become one of the most important research directions in data mining and machine learning. In multi-domain learning, one problem is that the classification patterns and data distributions are different among domains, which leads to that the knowledge (e.g. classification hyperplane) can not be directly transferred from one domain to another. This paper proposes a framework to combine class-separate objectives (maximize separability among classes) and domain-merge objectives (minimize separability among domains) to achieve cross-domain representation learning. Three special methods called DMCS_CSF, DMCS_FDA and DMCS_PCDML upon this framework are given and the experimental results valid their effectiveness.</p>",
    },	

    {
        id : "stf_neurocomputing",
        name : "A Subject Transfer Framework for EEG Classification. ",
		tags : new Array(tagList.ml, tagList.bci),
        coauthors : new Array(authorList.wttu, authorList.slsun),
        conference : "Neurocomputing",
        type : "j",
		year : "2014b",
        paperAbstract : "<p>This paper proposes a subject transfer framework for EEG classification. It aims to improve the classification performance when the training set of the target subject (namely user) is small owing to the need to reduce the calibration session. Our framework pursues improvement not only at the feature extraction stage, but also at the classification stage. At the feature extraction stage, we first obtain a candidate filter set for each subject through a previously proposed feature extraction method. Then, we design different criterions to learn two sparse subsets of the candidate filter set, which are called the robust filter bank and adaptive filter bank, respectively. Given robust and adaptive filter banks, at the classification step, we learn classifiers corresponding to these filter banks and employ a two-level ensemble strategy to dynamically and locally combine their outcomes to reach a single decision output. The proposed framework, as validated by experimental results, can achieve positive knowledge transfer for improving the performance of EEG classification.</p>",
    },	

    {
        id : "tddr_ictai",
        name : "Transferable Discriminative Dimensionality Reduction.  ",
        coauthors : new Array(authorList.wttu, authorList.slsun),
		tags : new Array(tagList.ml),
        conference : "Proceedings of the 23rd IEEE International Conference on Tools with Artificial Intelligence (ICTAI)",
        type : "c",
		year : "2014b",
        paperAbstract : "<p>In transfer learning scenarios, previous discriminative dimensionality reduction methods tend to perform poorly owing to the difference between source and target distributions. In such cases, it is unsuitable to only consider discrimination in the low-dimensional source latent space since this would generalize badly to target domains. In this paper, we propose a new dimensionality reduction method for transfer learning scenarios, which is called transferable discriminative dimensionality reduction (TDDR). By resolving an objective function that encourages the separation of the domain-merged data and penalizes the distance between source and target distributions, we can find a low-dimensional latent space which guarantees not only the discrimination of projected samples, but also the transferability to enable later classification or regression models constructed in the source domain to generalize well to the target domain. In the experiments, we firstly analyze the perspective of transfer learning in brain-computer interface (BCI) research and then test TDDR on two real datasets from BCI applications. The experimental results show that the TDDR method can learn a low-dimensional latent feature space where the source models can perform well in the target domain.</p>",
    },	

    {
        id : "sslbci_ijcnn",
        name : "Semi-supervised Feature Extraction with Local Temporal Regularization for EEG classification. ",
		tags : new Array(tagList.ml, tagList.bci),
        coauthors : new Array(authorList.wttu, authorList.slsun),
        conference : "Proceedings of the 21st International Joint Conference on Neural Networks (IJCNN)",
        type : "c",
		year : "2014b",
        paperAbstract : "<p>Extreme energy ratio (EER) is a recently proposed feature extractor to learn spatial filters for electroencephalogram (EEG) signal classification. It is theoretically equivalent and computationally superior to the common spatial patterns (CSP) method which is a widely used technique in brain-computer interfaces (BCIs). However, EER may seriously overfit on small training sets due to the presence of large noise. Moreover, it is a totally supervised method that cannot take advantage of unlabeled data. To overcome these limitations, we propose a regularization constraint utilizing local temporal information of unlabeled trails. It can encourage the temporal smoothness of source signals discovered, and thus alleviate their tendency to overfit. By combining this regularization trick with the EER method, we present a semi-supervised feature extractor termed semi-supervised extreme energy ratio (SEER). After solving two eigenvalue decomposition problems, SEER recovers latent source signals that not only have discriminative energy features but also preserve the local temporal structure of test trails. Compared to the features found by EER, the energy features of these source signals have a stronger generalization ability, as shown by the experimental results. As a nonlinear extension of SEER, we further present the kernel SEER and provide the derivation of its solutions.</p>",
    },	

    {
        id : "ieer_iconip",
        name : "Importance Weighted Extreme Energy Ratio for EEG Classification.",
		tags : new Array(tagList.ml, tagList.bci),
        coauthors : new Array(authorList.wttu, authorList.slsun),
        conference : "Proceedings of the 17th International Conference on Neural Information Processing (ICONIP)",
        type : "c",
		year : "2014b",
        paperAbstract : "<p>Spatial filtering is important for EEG signal processing since raw scalp EEG potentials have a poor spatial resolution due to the volume conduction effect. Extreme energy ratio (EER) is a recently proposed feature extractor which exhibits good performance. However, the performance of EER will be degraded by some factors such as outliers and the time-variances between the training and test sessions. Unfortunately, these limitations are common in the practical brain-computer interface (BCI) applications. This paper proposes a new feature extraction method called importance-weighted EER (IWEER) by defining two kinds of weight termed intra-trial importance and inter-trial importance. These weights are defined with the density ratio theory and assigned to the data points and trials respectively to improve the estimation of covariance matrices. The spatial filters learned by the IWEER are both robust to the outliers and adaptive to the test samples. Compared to the previous EER, experimental results on nine subjects demonstrate the better classification ability of the IWEER method.</p>",
    },	

    {
        id : "sfs_adma",
        name : "Spatial Filter Selection with Lasso for EEG Classification.",
		tags : new Array(tagList.ml, tagList.bci),
        coauthors : new Array(authorList.wttu, authorList.slsun),
        conference : "Proceedings of the 6th International Conference on Advanced Data Mining and Applications (ADMA)",
        type : "c",
		year : "2014b",
        paperAbstract : "<p>Spatial filtering is an important step of preprocessing for electroencephalogram (EEG) signals. Extreme energy ratio (EER) is a recently proposed method to learn spatial filters for EEG classification. It selects several eigenvectors from top and end of the eigenvalue spectrum resulting from a spectral decomposition to construct a group of spatial filters as a filter bank. However, that strategy has some limitations and the spatial filters in the group are often selected improperly. Therefore the energy features filtered by the filter bank do not contain enough discriminative information or severely overfit on small training samples. This paper utilize one of the penalized feature selection strategies called LASSO to aid us to construct the spatial filter bank termed LASSO spatial filter bank. It can learn a better selection of the spatial filters. Then two different classification methods are presented to evaluate our LASSO spatial filter bank. Their excellent performances demonstrate the stronger generalization ability of the LASSO spatial filter bank, as shown by the experimental results.</p>",
    },	


    {
        id : "mvssl_isnn",
        name : "View construction for multi-view semi-supervised Learning.",
		tags : new Array(tagList.ml),
        coauthors : new Array(authorList.slsun, authorList.fjin, authorList.wttu),
        conference : "Proceedings of the 8th International Symposium on Neural Networks (ISNN)",
        type : "c",
		year : "2014b",
        paperAbstract : "<p>Recent developments on semi-supervised learning have witnessed the effectiveness of using multiple views, namely integrating multiple feature sets to design semi-supervised learning methods. However, the so-called multi-view semi-supervised learning methods require the availability of multiple views. For many problems, there are no ready multiple views, and although the random split of the original feature sets can generate multiple views, it is definitely not the most effective approach for view construction. In this paper, we propose a feature selection approach to construct multiple views by means of genetic algorithms. Genetic algorithms are used to find promising feature subsets, two of which having maximum classification agreements are then retained as the best views constructed from the original feature set. Besides conducting experiments with single-task support vector machine (SVM) classifiers, we also apply multi-task SVM classifiers to the multi-view semi-supervised learning problem. The experiments validate the effectiveness of the proposed view construction method.</p>",
    },	

	
);
