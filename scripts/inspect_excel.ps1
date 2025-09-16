$ErrorActionPreference = 'Stop'
param(
  [Parameter(Mandatory=$false)][string]$Path = "数据_区域双碳目标与路径规划研究（含拆分数据表）.xlsx"
)

function Write-Header($text) {
  Write-Output ("=" * 80)
  Write-Output $text
  Write-Output ("=" * 80)
}

function Get-ZipEntryText([IO.Compression.ZipArchive]$zip, [string]$entryPath) {
  $entry = $zip.Entries | Where-Object { $_.FullName -eq $entryPath }
  if (-not $entry) { return $null }
  $sr = New-Object IO.StreamReader($entry.Open())
  try { return $sr.ReadToEnd() } finally { $sr.Dispose() }
}

function Get-Xml([string]$xmlText) {
  $doc = New-Object System.Xml.XmlDocument
  $doc.PreserveWhitespace = $true
  $doc.LoadXml($xmlText)
  return $doc
}

function Add-Ns([System.Xml.XmlDocument]$doc, [hashtable]$nsMap) {
  $nsm = New-Object System.Xml.XmlNamespaceManager($doc.NameTable)
  foreach ($k in $nsMap.Keys) { $nsm.AddNamespace($k, $nsMap[$k]) }
  return $nsm
}

function Get-SharedStrings($zip) {
  $sstText = Get-ZipEntryText $zip 'xl/sharedStrings.xml'
  if (-not $sstText) { return @() }
  $sstXml = Get-Xml $sstText
  $nsm = Add-Ns $sstXml @{ ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main' }
  $strings = @()
  foreach ($si in $sstXml.SelectNodes('//ns:si', $nsm)) {
    # Concatenate all descendant t nodes
    $tNodes = $si.SelectNodes('.//ns:t', $nsm)
    if ($tNodes -and $tNodes.Count -gt 0) {
      $text = ($tNodes | ForEach-Object { $_.InnerText }) -join ''
      $strings += ,$text
    } else {
      $strings += ,''
    }
  }
  return ,$strings
}

function Get-ColumnIndexFromRef([string]$cellRef) {
  if (-not $cellRef) { return 1 }
  if ($cellRef -match '^[A-Z]+') {
    $letters = $Matches[0]
    $sum = 0
    foreach ($ch in $letters.ToCharArray()) {
      $sum = $sum * 26 + ([int][char]$ch - [int][char]'A' + 1)
    }
    return $sum
  }
  return 1
}

function Inspect-Workbook([string]$xlsxPath) {
  if (-not (Test-Path -Path $xlsxPath)) {
    Write-Output "File not found: $xlsxPath"
    exit 1
  }
  Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
  $zip = [IO.Compression.ZipFile]::OpenRead($xlsxPath)
  try {
    $workbookText = Get-ZipEntryText $zip 'xl/workbook.xml'
    if (-not $workbookText) { throw "Not a valid .xlsx (missing xl/workbook.xml)" }
    $workbookXml = Get-Xml $workbookText
    $relsText = Get-ZipEntryText $zip 'xl/_rels/workbook.xml.rels'
    $relsXml = if ($relsText) { Get-Xml $relsText } else { $null }

    $nsm = Add-Ns $workbookXml @{
      ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
      r  = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    }
    $relsNsm = if ($relsXml) { Add-Ns $relsXml @{ rel = 'http://schemas.openxmlformats.org/package/2006/relationships' } } else { $null }

    # Build relationship map r:id -> Target
    $relMap = @{}
    if ($relsXml) {
      foreach ($rel in $relsXml.SelectNodes('//rel:Relationship', $relsNsm)) {
        $relMap[$rel.Attributes['Id'].Value] = $rel.Attributes['Target'].Value
      }
    }

    $sheets = @()
    foreach ($sheet in $workbookXml.SelectNodes('//ns:sheets/ns:sheet', $nsm)) {
      $rid = $sheet.Attributes['r:id'].Value
      $name = $sheet.Attributes['name'].Value
      $target = if ($relMap.ContainsKey($rid)) { $relMap[$rid] } else { "worksheets/sheet$($sheet.Attributes['sheetId'].Value).xml" }
      if (-not $target.StartsWith('xl/')) { $target = "xl/$target" }
      $sheets += [pscustomobject]@{ Name = $name; Target = $target }
    }

    Write-Header "Workbook: $(Split-Path $xlsxPath -Leaf) — Sheets: $($sheets.Count)"
    $sharedStrings = Get-SharedStrings $zip

    $sheetIndex = 0
    foreach ($s in $sheets) {
      $sheetIndex++
      $sheetText = Get-ZipEntryText $zip $s.Target
      if (-not $sheetText) {
        Write-Output ("[{0:d2}] Sheet: {1}" -f $sheetIndex, $s.Name)
        Write-Output "    (could not open sheet xml)"
        continue
      }
      $sheetXml = Get-Xml $sheetText
      $nsmSheet = Add-Ns $sheetXml @{ ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main' }
      $rows = $sheetXml.SelectNodes('//ns:sheetData/ns:row', $nsmSheet)
      $maxRow = if ($rows) { [int]($rows | Select-Object -Last 1).Attributes['r'].Value } else { 0 }

      # Build a simple function to get cell value
      function Get-CellValue($cellNode) {
        $tAttr = $cellNode.Attributes['t']
        $t = if ($tAttr) { $tAttr.Value } else { $null }
        if ($t -eq 's') {
          $vNode = $cellNode.SelectSingleNode('ns:v', $nsmSheet)
          if ($vNode -and $vNode.InnerText -ne '') {
            $idx = [int]$vNode.InnerText
            if ($idx -ge 0 -and $idx -lt $sharedStrings.Count) { return $sharedStrings[$idx] }
            else { return $vNode.InnerText }
          } else { return '' }
        } elseif ($t -eq 'inlineStr') {
          $tNodes = $cellNode.SelectNodes('ns:is//ns:t', $nsmSheet)
          if ($tNodes -and $tNodes.Count -gt 0) { return (($tNodes | ForEach-Object { $_.InnerText }) -join '') }
          else { return '' }
        } else {
          $vNode = $cellNode.SelectSingleNode('ns:v', $nsmSheet)
          return if ($vNode) { $vNode.InnerText } else { '' }
        }
      }

      # Gather header from the first row
      $header = @{}
      $headerValues = @()
      $firstRow = $rows | Select-Object -First 1
      if ($firstRow) {
        foreach ($c in $firstRow.SelectNodes('ns:c', $nsmSheet)) {
          $ref = $c.Attributes['r'].Value
          $colIdx = Get-ColumnIndexFromRef $ref
          $val = Get-CellValue $c
          $header[$colIdx] = $val
        }
        $maxColIdx = if ($header.Keys.Count -gt 0) { ($header.Keys | Measure-Object -Maximum).Maximum } else { 0 }
        for ($i = 1; $i -le $maxColIdx; $i++) { $headerValues += $(if ($header.ContainsKey($i)) { $header[$i] } else { '' }) }
      }

      Write-Output ("[{0:d2}] Sheet: {1}" -f $sheetIndex, $s.Name)
      Write-Output ("    Approx rows: {0}" -f $maxRow)
      if ($headerValues.Count -gt 0) {
        Write-Output ("    Columns ({0}): {1}" -f $headerValues.Count, ('[' + ($headerValues -join ', ') + ']'))
      } else {
        Write-Output "    Columns: (header row empty or not found)"
      }

      # Preview first two data rows after header
      $rowsEnum = $rows | Select-Object -Skip 1 | Select-Object -First 2
      if (-not $rowsEnum -or $rowsEnum.Count -eq 0) {
        Write-Output "    (no data rows)"
      } else {
        foreach ($r in $rowsEnum) {
          $rowCells = @{}
          foreach ($c in $r.SelectNodes('ns:c', $nsmSheet)) {
            $ref = $c.Attributes['r'].Value
            $colIdx = Get-ColumnIndexFromRef $ref
            $rowCells[$colIdx] = (Get-CellValue $c)
          }
          $values = @()
          $maxColIdx2 = if ($headerValues.Count -gt 0) { $headerValues.Count } else { if ($rowCells.Keys.Count -gt 0) { ($rowCells.Keys | Measure-Object -Maximum).Maximum } else { 0 } }
          for ($i = 1; $i -le $maxColIdx2; $i++) { $values += $(if ($rowCells.ContainsKey($i)) { $rowCells[$i] } else { '' }) }
          Write-Output ("    Row: [" + ($values -join ', ') + "]")
        }
      }
    }
  } finally {
    $zip.Dispose()
  }
}

Inspect-Workbook -xlsxPath $Path

