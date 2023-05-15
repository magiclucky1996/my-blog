# try hack me- penetration tester



## introduction t cyber security

- offensive security
  - find hidden page and hack the bank
  - breaking into system, exploit bug, find loopholes



## walk an application

- Short breakdown
  - view source
  - Network
  - Inspector: 
    - view the blocks
  - debugger
    - control JS
- explore the web
  - view web sites frame
- page source
  - most are made of **frame**
- inspector
  - view the css style and change
    - Display: block. -> display: none
- debugger
  - using break point to stop the running of the js
- Network

## content discovery

- **Robots.txt**





- **Favicon**

  - https://wiki.owasp.org/index.php/OWASP_favicon_database

  - look up the hash value of favicon

  - find the frame





- **Sitemap.xml**
  - list of files the owner wish to be listed on search engine





- **http headers**
  - Contains information like: 
    - 1. server software
      2. Programming language
  - visit by terminal: curl "ip address" -v



- **framework stack**
  - View the frame from the source information, 
  - find some drawback from the frame documentation

- **google hacking**
  - Site:tryhackme.com
  - Inurl: admin
  - filetype
  - Entitle: admin

- **Wappalyzer**
  - Analyze the frame that this page use
  - has google extension
- **wayback machine**
- github
  -  look for **company names** or **website names** to  locate repositories of target

- **S3 Buckets**
  - Cloud, sometims forget to set private
  - http(s)://**{name}.**[**s3.amazonaws.com**](http://s3.amazonaws.com/)
  - where to find the url:
    - Web source
    - github repo
    - Animated generate
      - **{name}**-assets, **{name}**-www, **{name}**-public, **{name}**-private
- **Automated Discovery**
  - Tools to automated
    - ffuf
    - dirb
    - Robuster
- conclusion
  - 步骤是可复用的，直接用脚本自动化就可以，



### Subdomain enumeate

- Ssl/tsl
  - find sub domain by ssl service look up
    - http://crt.sh/
    - https://ui.ctsearch.entrust.com/ui/ctsearchui
- google
  - **-site:www.tryhackme.com site:\*.tryhackme.com**
- DNS brute force（子域爆破）
- autamated tool
  - https://github.com/aboul3la/Sublist3r
- Virtual host(爆破host name，替换)
  - enumerate for host names on a server
  - when server host multiple webs on same ip, host name is used to identify





### authentication bypass

- User name enumerate
  - test which user name exists
- Brute force
- logic flaw
- Cookie tampering
  - Cookie  会进行编码

### idor

- detect
  - create two accounts, swap the id between them
- parameter may not be address
  - AJAX
  - JS
-  an unreferenced parameter that may have been of some use
  - **/user/details?user_id=123**
- practical
  - network 看到从api通过userid请求数据
  - 更改链接传参中的userid来获取数据