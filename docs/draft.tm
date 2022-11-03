<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Bounding the Failure Probability of Relu-Networks with
  Neuro-Symbolic Verification>>

  <abstract-data|<abstract|Neuro-symbolic verification has recently emerged
  as a new technique to verify high-level semantical properties of neural
  networks. The main idea is to use a set of trustworthy networks that
  express high-level properties which can be addressed as predicates in an
  otherwise logical specification. The approach relies on modern SMT solvers
  which allow to verify the network that is under consideration against the
  specification. Low-level properties such as adversarial robustness are
  expressible in this framework in the same way as high-level properties such
  as \Qa self-driving car will always hold in front of a stop sign'. In case
  of a violation of the property, the method is also able to provide concrete
  counter examples. However, searching for counterexamples in the entire
  feature space might return instances which are not in the support of the
  data distribution and therefore might be meaningless for the downstream
  task. Xie et al. propose an auto-encoder based method to ensure that
  counterexamples are contained in the data distribution. We build upon this
  idea and refine it to yield probabilisticly interpretable results. We
  achieve this by replacing the autoencoder with a density estimator and
  adjust the verfication task to verify properties only for the upper level
  density set of a specified probability mass. This allows an intuitive
  probabilistiv interpretation of the varification result. We investigate
  consistency, encoding, and approximation related properties for kernel
  density estimators and affine autoregressive flows.>>

  <section|Introduction>

  Goal of this paper to build upon the idea from
  <cite|xie_neurosymbolic_2022> of restricting neuro-symbolic verfication
  tasks to the support of the data distribution. We think that this is a very
  important feature for verification of real world systems since in practice
  we can often not expect that the model behaves well in an unlikely area of
  the input space where little to no data is available. In fact, such an area
  might even be meaningless. However, one would like to ensure that
  misbehavior can only happen in such unlikely areas. In
  <cite|xie_neurosymbolic_2022> an autoencoder is used to define the support
  for the verification task based on the reconstruction error. While
  autoencoder implicitly capture the data distribution through the
  reconstruction error, they are not trained to align the reconstruction
  error with the underlying density function. Hence, it is not necessary the
  case that high reconstruction error means low density. For instance, an
  autoencoder trained on images that depict objects of various shapes might
  produce a larger reconstruction error on the more complex shapes even if
  they are in very high density areas with respect to the data distribution.
  If <math|D> is the data distribution with density <math|p<rsub|D>>, then
  the upper density level set for a threshold <math|t>,
  <math|L<rsup|\<uparrow\>><rsub|D><around*|(|t|)>=<around*|{|x\<in\>\<bbb-R\><rsup|d>\<mid\>p<rsub|D><around*|(|x|)>\<gtr\>t|}>>,
  is a natural replacement which allows a probabilistic interpretation of the
  support. The probability of the occurrence of a misbehavior is obviously
  upper bounded by the probability of the corresponding lower density level
  set <math|L<rsup|\<downarrow\>><rsub|D><around*|(|t|)>=<around*|{|x\<in\>\<bbb-R\><rsup|d>\<mid\>p<rsub|D><around*|(|x|)>\<leq\>t|}>>.
  In a practical scenario, we would prefer to specify acceptable failure
  probability <math|p> rather than an acceptable density threshold. Hence, we
  propose the following abstract procedure for verification of machine
  learning models within the support of the data distribution:

  <\enumerate-numeric>
    <item>For a given <math|p\<in\><around*|[|0,1|]>>, determine the density
    <math|t<rsub|p>> with <math|p<rsub|D><around*|(|L<rsup|\<downarrow\>><rsub|D><around*|(|t<rsub|p>|)>|)>=p>.

    <item>Verify that <math|\<forall\>x:p<rsub|D><around*|(|x|)>\<gtr\>t<rsub|p>\<rightarrow\>\<varphi\><around*|(|x|)>>
  </enumerate-numeric>

  Where <math|\<varphi\>> is the neuro-symbolic property that we want to
  verify. We might also choose to work with <math|log p<rsub|D>> instead of
  <math|p<rsub|D>> whenever this is more convenient. In case that the
  verification succeeds, we have a guarantee that a misbehavior can only
  occur in the <math|p>-th most unlikely fraction of the data. This holds of
  course only if we have access to the true density finction. In most cases,
  however, we won't have access to <math|p<rsub|D>> and we will need to
  estimate it from data. We investigate wether there is a consistent
  procedure using plug-in estimates that will almost surely produce the
  correct answer in the limit case of infinite data supply. The convergence
  rate of the plug-in estimate usually depends exponentially on the data
  dimension. Therefore, we will also explore deep generative modeling of
  high-dimensional density functions using affine autoregressive flows. In
  both cases, we will have to ensure that the (log-)density function can be
  appropriately encoded for the verification procedure.

  <section|Density Estimators>

  <subsection|Kernel Density Estimation>

  Non-parametric density estimation goes back to the early work of Parzen and
  Rosenblatt in the 1950s. A kernel density estimator spreads an equal amount
  of the available probability mass around the points in the training set.
  The distribution of the mass is governed by a normalized symmetric kernel
  function <math|K:<with|font|Bbb|R><rsup|d>\<rightarrow\>\<bbb-R\><rsup|>>.
  A kernel is radial if <math|K<around*|(|x|)>=k<around*|(|<around*|\||x|\|>|)>>
  for some function <math|k:\<bbb-R\>\<rightarrow\>\<bbb-R\>>. Given a
  bandwidth parameter <math|h\<in\>\<bbb-R\><rsub|+><rsup|d>> the kernel
  <math|K<rsub|>> with bandwidth <math|h> is
  <math|K<rsub|h><around*|(|x|)>=<frac|1|<big|prod><rsub|i>h<rsub|i><rsup|>>K<rsub|><around*|(|<around*|(|<frac|x<rsub|1>|h<rsub|<rsub|1>>>,\<ldots\>.,<frac|x<rsub|n>|h<rsub|n>>|)><rsup|T>|)>>.
  Given a dataset <math|X=<around*|{|x<rsub|1>,\<ldots\>,x<rsub|k>|}>> that
  was drawn independently from an unknown distribution <math|P>, the kernel
  density estimate (KDE) with kernel <math|K> and bandwidth <math|h> is
  <math|<wide|p|^><around*|(|x|)>=<frac|1|n><big|sum><rsub|i=1><rsup|k>K<rsub|h><around*|(|x-x<rsub|i>|)>>.

  <subsection|Normalizing Flows>

  Normalizing flows transform the target distribution <math|D> through a
  continuous invertible map <math|\<phi\><rsub|\<theta\>>> into a simple base
  distribution <math|B>. The map <math|\<phi\><rsub|\<theta\>>> is
  represented by an invertible neural network with parameters
  <math|\<theta\>>. Normalizing flows have the property that they can allow
  for both, efficient sampling and efficient computation of log-likelihoods:

  <\enumerate>
    <item>Sampling is performed by first sampling <math|z> from the <math|B>
    and then computing the inverse map <math|\<phi\><rsub|\<theta\>><rsup|-1><around*|(|z|)>>.

    <item>Computing the likelihood for some <math|x> is done by first
    computing the likelihood of <math|\<phi\><rsub|\<theta\>><around*|(|x|)>>
    under the base distribution and then applying the change of variables
    formula

    <\eqnarray*>
      <tformat|<table|<row|<cell|p<rsub|D><around*|(|x|)>>|<cell|=>|<cell|<around*|\||<frac|\<partial\>\<phi\><rsub|\<theta\>>|\<partial\>x<rsup|T>><around*|(|x|)>|\|>
      <rsup|>p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>,>>>>
    </eqnarray*>
  </enumerate>

  Usually, neural networks do not represent bijections and one therefore
  needs to design specific architectures that ensure bijectivity. This is
  usually done by ensuring bijectivity layer-wise.\ 

  <subsubsection|Affine Autoregressive Flows>

  Affine autoregressive flows are among the simplest flow architectures. They
  are composed from affine autoregressive layers, which compute a function
  <math|\<psi\><rsub|\<theta\>><around*|(|x|)>=z> with
  <math|z<rsub|i>=a<rsub|i>x<rsub|i>+b<rsub|i>> and
  <math|<around*|(|a<rsub|i>,b<rsub|i>|)>=c<rsub|i,\<theta\>><around*|(|z<rsub|1>,\<ldots\>,z<rsub|i-1>|)>>.
  The functions <math|c<rsub|i,\<theta\>>> are called conditioners because
  they encode the dependency of <math|z<rsub|i>> on
  <math|z<rsub|1>,\<ldots\>z<rsub|i-1>>. As long as <math|a<rsub|i>\<neq\>0>
  for all <math|i>, the function <math|\<psi\><rsub|\<theta\>>> is bijective:
  Indeed, <math|<around*|\||<frac|\<partial\>\<phi\><rsub|\<theta\>>|\<partial\>x<rsup|T>><around*|(|x|)>|\|>\<neq\>0>
  for all <math|x> since <math|z<rsub|i>> does not depend on <math|x<rsub|j>>
  for <math|i\<less\>j>. Therefore, <math|<frac|\<partial\>\<phi\><rsub|\<theta\>>|\<partial\>x<rsup|T>><around*|(|x|)>>
  is lower triangular and with that <math|<around*|\||<frac|\<partial\>\<phi\><rsub|\<theta\>>|\<partial\>x<rsup|T>><around*|(|x|)>|\|>=<op|Tr><around*|(|<frac|\<partial\>\<phi\><rsub|\<theta\>>|\<partial\>x<rsup|T>><around*|(|x|)>|)>=<big|prod><rsub|i>a<rsub|i>\<neq\>0>.
  \ 

  In the following let <math|\<phi\><rsub|\<theta\>>=\<psi\><rsub|n,\<theta\><rsub|>>\<circ\>\<cdots\>\<circ\>\<psi\><rsub|1,\<theta\>>>
  be composed of affine autoregressive layers
  <math|\<psi\><rsub|i,\<theta\>>> each with conditioners <math|c<rsub|i
  j,\<theta\>>>. We use <math|\<phi\><rsub|i,\<theta\>>\<assign\>\<psi\><rsub|i,\<theta\><rsub|>>\<circ\>\<cdots\>\<circ\>\<psi\><rsub|1,\<theta\>>>
  for <math|1\<leq\>i\<leq\>n> and denote the identity
  <math|\<phi\><rsub|0,\<theta\>>>. In order to compute the log-likelihood,
  one uses the decomposition in the change of variables formula:

  <\eqnarray*>
    <tformat|<table|<row|<cell|log p<around*|(|x|)>>|<cell|=>|<cell|log<around*|(|<around*|\||<frac|\<partial\>\<phi\><rsub|\<theta\>>|\<partial\>x<rsup|T>><around*|(|x|)>|\|><rsup|>p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|log
    p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>+log<around*|(|<around*|\||<frac|\<partial\>\<phi\><rsub|\<theta\>>|\<partial\>x<rsup|T>><around*|(|x|)>|\|><rsup|>|)>>>|<row|<cell|>|<cell|=>|<cell|log
    p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>+<rsub|>log<around*|(|<big|prod><rsup|n><rsub|i=1><big|prod><rsub|j=1><rsup|d>a<rsub|i
    j>|)>>>|<row|<cell|>|<cell|=>|<cell|log
    p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>+<rsub|><around*|(|<big|sum><rsup|n><rsub|i=1><big|sum><rsub|j=1><rsup|d>log
    a<rsub|i j>|)>>>>>
  </eqnarray*>

  \ where <math|a<rsub|i j>=c<rsub|i j,\<theta\>><around*|(|\<phi\><rsub|i,\<theta\>><around*|(|x|)><rsub|:j-1>|)><rsub|0>>.
  In the following we will assume that conditioner are build such that
  <math|a<rsub|i j>\<gtr\>c\<gtr\>0> for some small constant <math|c>.

  <section|Encoding Density Estimators>

  <\lemma>
    Let <math|K> be an encodable kernel and
    <math|X<rsup|>=<around*|{|x<rsub|1>,\<ldots\>,x<rsub|k>|}>> a data set.
    Then the kernel density estimate with kernel <math|K> trained on <math|X>
    is encodable. \ 
  </lemma>

  <\proof>
    If <math|K> is encodable then <math|<wide|p|^><around*|(|x|)>=<frac|1|n><big|sum><rsup|><rsub|<rsub|i>>K<around*|(|x-x<rsub|i>|)>>
    is obviously also encodable.
  </proof>

  Note that many commonly used kernel such as the Gaussian kernel are not
  directly encodable. However, there are also some common kernels which allow
  obvious encodings:

  <\itemize-dot>
    <item>The top-hat kernel <math|K<rsub|<text|uniform>><around*|(|x|)>=<choice|<tformat|<table|<row|<cell|<frac|1|2<rsup|d>><text|
    ; if >x\<in\><around*|[|-1,1|]><rsup|d>>>|<row|<cell|0<space|1em><text|;
    else>>>>>>>

    <item>The linear kernel under <math|L<rsub|1>>-norm
    <math|K<rsub|<text|<math|L<rsub|1><text|-linear>>>><around*|(|x|)>=<choice|<tformat|<table|<row|<cell|<frac|d|<rsup|<around*|(|<sqrt|2>|)><rsup|d>>><around*|(|1-<big|sum><rsub|i><around*|\||x<rsub|i>|\|>|)><text|
    ; if ><big|sum><rsub|i><around*|\||x<rsub|i>|\|>\<leq\>1>>|<row|<cell|0<space|1em><text|;
    else>>>>>>>
  </itemize-dot>

  <\lemma>
    Let <math|\<phi\><rsub|\<theta\>>> be an affine autoregressive flow with
    ReLu conditioning networks over a feasible base distribution. Then for
    any <math|\<epsilon\>\<gtr\>0> there is an encoding of an
    <math|\<epsilon\>>-approximation of <math|\<phi\><rsub|\<theta\>>> over
    any bounded domain <math|X\<subseteq\>\<bbb-R\><rsup|d>>.\ 
  </lemma>

  <\proof>
    Let <math|n> be the number of layers of <math|\<phi\><rsub|\<theta\>>>
    and let <math|c<rsub|<rsub|i j,\<theta\>>>> denote the conditioners of
    layer <math|i>. Define <math|C\<assign\><around*|{|c<rsub|<rsub|i
    j,\<theta\>>><around*|(|\<phi\><rsub|i,\<theta\>><around*|(|x|)><rsub|:j-1>|)><rsub|0>\<mid\>i\<leq\>n,j\<leq\>d,x\<in\>X|}>\<subseteq\>\<bbb-R\>>.
    Since <math|X> is bounded, <math|C> is also bounded because
    <math|\<phi\><rsub|\<theta\>>> is build from ReLu-networks and affine
    transformations. Also, <math|inf C> \<geq\> c for some <math|c\<gtr\>0>
    by our assumption on the conditioners. Hence, the exists an encodable
    <math|<around*|(|<frac|\<epsilon\>|n d>|)>>-approximation of the natural
    logarithm <math|\<alpha\><rsub|log>> over <math|C>. As previously seen,
    the conditioning networks and the flow network are encodable. By
    assumption, the log-likelihood function of the base distribution is also
    encodable, and hence also <math|log p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>>.
    We can now combine them through a linear combination and see that the
    function <math|\<alpha\><around*|(|x|)>\<assign\>log
    p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>-<rsub|><around*|(|<big|sum><rsup|n><rsub|i=1><big|sum><rsub|j=1><rsup|d>\<alpha\><rsub|log><around*|(|a<rsub|i
    j>|)>|)>> is encodable. We claim that <math|\<alpha\>> is an
    <math|\<epsilon\>>-approximation of <math|log p<rsub|D>> on <math|X>.
    Indeed, for <math|x\<in\>X>:

    <\eqnarray*>
      <tformat|<table|<row|<cell|<around*|\||p<rsub|D><around*|(|x|)>-\<alpha\><around*|(|x|)>|\|>>|<cell|=>|<cell|<around*|\||log
      p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>+<rsub|><around*|(|<big|sum><rsup|n><rsub|i=1><big|sum><rsub|j=1><rsup|d>log
      a<rsub|i j>|)>-<around*|(|log p<rsub|B><around*|(|\<phi\><rsub|\<theta\>><around*|(|x|)>|)>+<around*|(|<big|sum><rsup|n><rsub|i=1><big|sum><rsub|j=1><rsup|d>\<alpha\><rsub|log><around*|(|a<rsub|i
      j>|)>|)>|)>|\|>>>|<row|<cell|>|<cell|\<leq\>>|<cell|<big|sum><rsup|n><rsub|i=1><big|sum><rsub|j=1><rsup|d><around*|\||log
      a<rsub|i j>-\<alpha\><rsub|log><around*|(|a<rsub|i
      j>|)>|\|>>>|<row|<cell|>|<cell|\<leq\>>|<cell|<big|sum><rsup|n><rsub|i=1><big|sum><rsub|j=1><rsup|d><frac|\<epsilon\>|n
      d>>>|<row|<cell|>|<cell|=>|<cell|\<epsilon\>>>>>
    </eqnarray*>

    where <math|a<rsub|i j>=c<rsub|i j,\<theta\>><around*|(|\<phi\><rsub|i,\<theta\>><around*|(|x|)><rsub|:j-1>|)><rsub|0>>.
    The second inequality holds because the <math|a<rsub|i j>> will be
    contained in <math|C> for any <math|x\<in\>X> by construction.
  </proof>

  <\bibliography|bib|tm-plain|TransferLab>
    <\bib-list|5>
      <bibitem*|1><label|bib-alvarez_estimation_2014>Diego<nbsp>A.<nbsp>Alvarez,
      Jorge<nbsp>E.<nbsp>Hurtado<localize|, and >Felipe Uribe.
      <newblock>Estimation of the Lower and Upper Probabilities of Failure
      Using Random Sets and Subset Simulation. <newblock><localize|Pages
      >905\U914, jul 2014.<newblock>

      <bibitem*|2><label|bib-kobyzev_normalizing_2020>Ivan Kobyzev,
      Simon<nbsp>J.<nbsp>D.<nbsp>Prince<localize|, and
      >Marcus<nbsp>A.<nbsp>Brubaker. <newblock>Normalizing Flows: An
      Introduction and Review of Current Methods.
      <newblock><with|font-shape|italic|ArXiv:1908.09257 [cs, stat]>, apr
      2020.<newblock>

      <bibitem*|3><label|bib-papamakarios_normalizing_2019>George
      Papamakarios, Eric Nalisnick, Danilo<nbsp>Jimenez Rezende, Shakir
      Mohamed<localize|, and >Balaji Lakshminarayanan. <newblock>Normalizing
      Flows for Probabilistic Modeling and Inference.
      <newblock><with|font-shape|italic|ArXiv:1912.02762 [cs, stat]>, dec
      2019.<newblock>

      <bibitem*|4><label|bib-tsybakov_nonparametric_1997>A.<nbsp>B.<nbsp>Tsybakov.
      <newblock>On Nonparametric Estimation of Density Level Sets.
      <newblock><with|font-shape|italic|The Annals of Statistics>,
      25(3):948\U969, 1997.<newblock>

      <bibitem*|5><label|bib-xie_neurosymbolic_2022>Xuan Xie, Kristian
      Kersting<localize|, and >Daniel Neider. <newblock>Neuro-Symbolic
      Verification of Deep Neural Networks. <newblock>mar 2022.<newblock>
    </bib-list>
  </bibliography>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|2>>
    <associate|auto-3|<tuple|2.1|2>>
    <associate|auto-4|<tuple|2.2|2>>
    <associate|auto-5|<tuple|2.2.1|2>>
    <associate|auto-6|<tuple|3|3>>
    <associate|auto-7|<tuple|2|3>>
    <associate|bib-alvarez_estimation_2014|<tuple|1|3>>
    <associate|bib-kobyzev_normalizing_2020|<tuple|2|3>>
    <associate|bib-papamakarios_normalizing_2019|<tuple|3|3>>
    <associate|bib-tsybakov_nonparametric_1997|<tuple|4|3>>
    <associate|bib-xie_neurosymbolic_2022|<tuple|5|3>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      xie_neurosymbolic_2022

      xie_neurosymbolic_2022
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Introduction>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Density
      Estimators> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Kernel Density Estimation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Normalizing Flows
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|2tab>|2.2.1<space|2spc>Affine Autoregressive
      Flows <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Encoding
      Density Estimators> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>